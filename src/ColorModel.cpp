
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