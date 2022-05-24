
#include "ColorModel.h"
#include <fstream>
#include <iostream>


//this function reads a mat from a .txt file and reshapes for a given number of rows
cv::Mat ReadMatFromTxt(std::string filename, int rows, int cols) 
{
	std::ifstream in(filename);
	std::vector<double> nums;
	while (in.good()){
		double n;
		in >> n;
		if (in.eof()) break;
		nums.push_back(n);
	}
	// now make a Mat from the vector:
	cv::Mat mat = cv::Mat(nums);
	cv::Mat mat_re = mat.reshape(1, rows); //reshape it so it has the correct number of rows/cols
	cv::Mat mat_clone = mat_re.clone(); // must clone it to return it, other wise it return a pointer to the values nums
	return mat_clone;
}

//this function writes a mat to file and is used for debugging
void writeMatToFile(cv::Mat& m, const char* filename) 
{
	std::ofstream fout(filename);

	if (!fout)
	{
		std::cout << "File Not Opened" << std::endl;  return;
	}

	for (int i = 0; i < m.rows; i++)
	{
		for (int j = 0; j < m.cols; j++)
		{
			fout << m.at<double>(i, j) << "\t";
		}
		fout << std::endl;
	}

	fout.close();
}


/**
*Compute 2D mahalanobis distance when color and distributions are projected to plane. See Eq 16 of "Unmixing-Based Soft Color Segmentation for Image Manipulation"
*/
double mg_computeMahalProj(cv::Vec3d p, cv::Vec3d normal, cv::Matx33d cov, cv::Vec3d color){
	// We need to project 'covariance' and 'color' to the plane defined by 'p' and 'normal'. To do this, we transform everything so that 'p' is at the origin and 'normal' is aligned with the z-axis.
	// Then, instead of projecting to the plane defined by 'p' and 'normal', we project everything onto the plane created by the x and y axis. (This makes the covariance projection easier).    

	//create rotation matrix to align normal to the z-axis. See https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
	cv::Mat nn, F_1;
	cv::Matx33d F_2, U_trans;

	//set up the variables needed to create G, F and U (see link)
	cv::normalize(cv::Mat(normal), nn);
	cv::Mat z = cv::Mat(cv::Vec3d(0,0,1)); //z axis
	cv::Mat v = nn.cross(z);
	double s = cv::norm(v);
	double c = nn.dot(z);
	cv::Mat v_mat = (z - c*nn)/cv::norm(z - c*nn);
	cv::Mat w_mat = z.cross(nn);
	
	//create G, F and U matrices as defined in link above.
	cv::Matx33d G = cv::Matx33d(c, -s, 0, s, c, 0 , 0, 0, 1);
	cv::hconcat(nn, v_mat, F_1);
	cv::hconcat(F_1, w_mat, F_2);
	cv::Matx33d F = F_2.inv();
	cv::Matx33d U = ((F.inv())*G*F);
	cv::transpose(cv::Mat(U), U_trans);

	//Transform color - first translate by p (as p has been moved to the origin) and rotate by U (because we aligned 'normal' to the z-axis).
	cv::Vec3d c_rot = U*(color - p);
	cv::Vec2d c_proj = cv::Vec2d(c_rot.val[0], c_rot.val[1]); //project transformed color onto xy plane

	cv::Mat cov1_rot = cv::Mat(U*cov*U_trans); //transform the covariance using the rotation U (see https://robotics.stackexchange.com/questions/2556/how-to-rotate-covariance)
	cv::Matx22d cov1_proj = cv::Matx22d(cov1_rot.at<double>(0,0),cov1_rot.at<double>(0,1),cov1_rot.at<double>(1,0),cov1_rot.at<double>(1,1)); //project to xy plane by removing 3rd dimension
	double mahal = std::pow(cv::Mahalanobis(c_proj, cv::Vec2d(0,0),(cov1_proj).inv()),2); // compute mahalanobis distance

	return mahal;
}

/**
* Projected Color Unmixing Energy. See Eq. 16 of "Unmixing-Based Soft Color Segmentation for Image Manipulation"
*/
double mg_projectedCUEnergyAlt(cv::Vec3d m1, cv::Vec3d m2, cv::Matx33d c1, cv::Matx33d c2, cv::Vec3d color){
	
	double cost = 0;

	//Check to make sure 'color' lies between the two planes defined by m1 and m2 
	cv::Vec3d n = (m1 - m2)/cv::norm(cv::Mat(m1 - m2));
	double dist = cv::norm(cv::Mat(m1 - m2)); // d(plane1, plane2)
	double d1 = cv::norm((color-m1).dot(n)); //d(color, plane1)
 	double d2 = cv::norm((color-m2).dot(n)); //d(color, plane2)
	if(d1 > dist || d2 > dist){

		cost = 100000; //if not between planes, set very high cost.

	}
	else{

		//Compute F (Eq 16 in unmixing paper)
		cv::Vec3d u1 = color - (((color - m1).dot(n))/(n.dot(n)))*n; //Eq 14 in unmixing paper
		cv::Vec3d u2 = color - (((color - m2).dot(n))/(n.dot(n)))*n;
		double alpha_1 = cv::norm(color - u2)/cv::norm(u1 - u2);//Eq 15 in unmixing paper
		double alpha_2 = 1 - alpha_1;
		double mahal1 = mg_computeMahalProj(m1, cv::Vec3d(m2-m1), c1, color); //computed projected mahalanobis distance
		double mahal2 = mg_computeMahalProj(m2, cv::Vec3d(m1-m2), c2, color);
		cost = alpha_1*(mahal1) + alpha_2*(mahal2); 
	}
	return cost;
}


/**
 * Representation score of input color. If the score is lower, the color is better represented with the current color model.
 */
double representationScoreAlt(cv::Vec3d color, int n, std::vector<cv::Vec3d> &means, std::vector<cv::Matx33d> &covs, double tau, int &proj){
	if(n == 0){ 
		// No colors yet in color model, all colors not well represented
		return std::pow(tau,2) + 1;

	} else if(n == 1){

		return std::pow(cv::Mahalanobis(color, means.at(0), (covs.at(0)).inv()),2);

	} else {

		std::vector<double> repList;
		std::vector<int> indList; //use this to check if a given color uses the projected color unmixing or the original color unmixing cost 
		double proj_cost, norm_cost;
		for(unsigned int i = 0; i < n; i++){

			norm_cost = (std::pow(cv::Mahalanobis(color, means.at(i), (covs.at(i)).inv()),2));
			repList.push_back(norm_cost);
			indList.push_back(0);
			for(unsigned int j = i+1; j < n; j++){ 

				proj_cost = mg_projectedCUEnergyAlt(means.at(i), means.at(j), covs.at(i), covs.at(j), color); //compute projected unmixing energy
				repList.push_back(proj_cost);
				indList.push_back(1);
			}
		}
		//use proj to see what colors are using projected unmixing - for debugging
		double min_cost = *min_element(repList.begin(), repList.end());
		for(unsigned int j = 0; j < repList.size(); j++){
			if(min_cost == repList.at(j)){
				proj = indList.at(j);
			}
		}
		return min_cost;
	}
}

/**
 * Get the bin that pixel c is voting for
 */
std::tuple<int, int, int> getVotingBinAlt(cv::Vec3d c){
	std::tuple<int, int, int> bin;
	bin = std::make_tuple(std::floor(c.val[0]/0.1),std::floor(c.val[1]/0.1),std::floor(c.val[2]/0.1));
	if(std::get<0>(bin) == 10){
			std::get<0>(bin) = 9;
		}
	if(std::get<1>(bin) == 10){
			std::get<1>(bin) = 9;
		}
	if(std::get<2>(bin) == 10){
			std::get<2>(bin) = 9;
		}

	return bin;
}

/**
 * Create guided filter kernel around the seed pixel at coordinates m,n
 * See Eq. 11 of "Guided Image Filtering" by He et. al.
 */
cv::Mat kernelValuesAlt(cv::Rect roi, cv::Mat &img, int m, int n){
	// Neighbourhood around seed pixel, each pixel gets a kernel value
	int count3;
	cv::Mat neighbourhood = img(roi).clone();
	//imwrite("kernel_neigh.png", neighbourhood*255);
	cv::Mat kernel(neighbourhood.size(), CV_64FC3, cv::Scalar(0.0));
	cv::Mat kernel_weights(neighbourhood.size(), CV_64FC1, cv::Scalar(0.0));
	cv::Vec3d S = img.at<cv::Vec3d>(m,n); // Seed pixel
	double eps = 0.01;
	double weight = 0;
	// Create filter window for every pixel in neighbourhood
	for(int i = roi.y; i < neighbourhood.rows+roi.y; i++){
		for(int j = roi.x; j < neighbourhood.cols+roi.x; j++){
			cv::Rect roiN(std::max(j-10,0),std::max(i-10,0),std::min(20,img.cols-j),std::min(20,img.rows-i));
			cv::Mat patch = img(roiN).clone();
			cv::Scalar nMean;
			cv::Scalar nStddev;
			cv::meanStdDev(patch, nMean, nStddev);
			// For every pixel in window, calculate value to add to kernel value
			for(int k = roiN.y; k < patch.rows+roiN.y; k++){ // k start patch corner (in image coord), goes to patch corner + 20
				for(int l = roiN.x; l < patch.cols+roiN.x; l++){ // l start patch corner (in image coord), goes to patch corner + 20
					int x = k - (m-10); //m-10 is where original patch start, if k is smaller, its outside original patch //coordinates in original patch
					int y = l - (n-10);//n-10 is where original patch start, if l is smaller, its outside original patch //coordinates in original patch
					if(x >= 0 && y >= 0 && x < 20 && y < 20){ // But only if that pixel was in the original neighbourhood
						cv::Vec3d I = img.at<cv::Vec3d>(k,l);
						double v1 = 1 + (((I.val[0] - nMean.val[0]) * (S.val[0] - nMean.val[0])) / (std::pow(nStddev.val[0],2) + eps));
						double v2 = 1 + (((I.val[1] - nMean.val[1]) * (S.val[1] - nMean.val[1])) / (std::pow(nStddev.val[1],2) + eps));
						double v3 = 1 + (((I.val[2] - nMean.val[2]) * (S.val[2] - nMean.val[2])) / (std::pow(nStddev.val[2],2) + eps));
						kernel.at<cv::Vec3d>(x,y) = kernel.at<cv::Vec3d>(x,y) + cv::Vec3d(v1,v2,v3);
					}
				}
			}
			//writeMatToFile(kernel,"kernel.txt");
			//std::string var_ker_s = std::string("kernel_") + std::to_string(i) + std::to_string(j) + std::string(".txt");
			//const char* cvec = var_ker_s.c_str();
			//writeMatToFile(kernel,cvec);

		}
	}
	for(size_t i = 0; i < kernel.rows; i++){
		for(size_t j = 0; j < kernel.cols; j++){
			weight = sqrt(kernel.at<cv::Vec3d>(i,j)[0]*kernel.at<cv::Vec3d>(i,j)[0] + kernel.at<cv::Vec3d>(i,j)[1]*kernel.at<cv::Vec3d>(i,j)[1] + kernel.at<cv::Vec3d>(i,j)[2]*kernel.at<cv::Vec3d>(i,j)[2]);
			kernel_weights.at<double>(i,j) = weight;
		}
	}
	return kernel_weights;
}


/**
 * Voting energy of pixel - eq 10 unmixing paper
*/
double getVoteAlt(cv::Vec3d gradient, double repScore){
	return std::pow(std::exp(-cv::norm(gradient, cv::NORM_L2)),2) * (1-std::exp(-repScore)); //mg added the square term 
}

/**
 * Calculate from which bin the next seed pixel should come from. See Section 5 of
 * "Unmixing-Based Soft Color Segmentation for Image Manipulation"
 */
std::tuple<int, int, int> nextBin(cv::Mat &img, cv::Mat &gradient, std::vector<cv::Vec3d> &means,
	std::vector<cv::Matx33d> &covs, cv::Mat &votemask, double tau, double &vote){
	
	//mg used for debugging to save image output
	//cv::Mat check_rep = cv::Mat(img.rows, img.cols,CV_64FC3, cv::Scalar(0.0,0.0,0.0)); //use this to check whether color uses proj unmixing or not