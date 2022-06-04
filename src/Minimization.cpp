
#include "Minimization.h"

using InputParams = std::tuple<std::vector<cv::Vec3d>, std::vector<cv::Matx33d>, cv::Vec4d, cv::Vec3d, double,
	bool, std::vector<double>, std::vector<double>>; //mg added this last term to store gt alphas for refinement


/**
 * Print vector v in cout
 */
void print_v(std::vector<double> v){
	for(double d : v){
		std::cout << d << " ";
	}
	std::cout << std::endl;
}

/**
 * Energy function of vector v. See Eq. 4 in "Unmixing-Based Soft Color Segmentation for Image Manipulation"
 * v: Input vector
 * means: mean values of color model
 * covs: inverse covariances of color model
 * sparse: true if the sparsity part should be included
 */
double energy(std::vector<double> v, std::vector<cv::Vec3d> &means, std::vector<cv::Matx33d> &covs, bool sparse){
	int n = means.size();
	double energy = 0;
	double dist, c1, c2, c3, alpha, sparsity;
	cv::Vec3d color;
	std::vector<double> alphas;
	for(size_t i = 0; i < n; i++){
		alpha = v.at(i);
		alphas.push_back(alpha);
		c1 = v.at(3*i+n);
		c2 = v.at(3*i+n+1);
		c3 = v.at(3*i+n+2);
		color = cv::Vec3d(c1,c2,c3);
		dist = pow(cv::Mahalanobis(color, means.at(i), (covs.at(i)).inv()),2);
		energy += alpha * dist;
	}
	if(sparse){
		double sum_alpha = 0.0;
		double sum_squared = 0.0;
		for(auto& n : alphas){
			sum_alpha += n;
			sum_squared += pow(n,2);
		}
		if(sum_squared == 0)
		{
			sparsity = 500;
		}
		else{
			sparsity = constants::sigma * (((sum_alpha)/(sum_squared)) - 1);
		}
	} else {
		sparsity = 0;
	}
	return energy + sparsity;
}

/**
 * Constraint vector g in the unmixing step. See Eq. 4 in "Interactive High-Quality Green-Screen Keying via Color Unmixing"
 * v: Input vector
 * n: number of colors in color model
 * color: Color of pixel in the input image
 */
cv::Vec4d g(std::vector<double> &v, int n, cv::Vec3d color){
	double g1 = 0, g2 = 0, g3 = 0, g4 = 0;
	for(size_t i = 0; i < n; i++){
		g1 += v.at(i)*v.at(3*i+n);
		g2 += v.at(i)*v.at(3*i+n+1);
		g3 += v.at(i)*v.at(3*i+n+2);
		g4 += v.at(i);
	}
	return cv::Vec4d(pow(g1-color.val[0],2),pow(g2-color.val[1],2),pow(g3-color.val[2],2),pow(g4-1,2));
}

/**
 * Partial derivative of g in respect to alpha
 * v: minimization vector
 * n: number of layers
 * color: color of pixel
 * k: index of alpha value for this partial derivative
 */
cv::Vec4d dg_a(std::vector<double> &v, int n, cv::Vec3d color, int k){//mg checked - seems correct
	double g1 = 0, g2 = 0, g3 = 0, g4 = 0;
	for(size_t i = 0; i < n; i++){
		g1 += v.at(i)*v.at(3*i+n);//r
		g2 += v.at(i)*v.at(3*i+n+1);//g
		g3 += v.at(i)*v.at(3*i+n+2);//b
		g4 += v.at(i);//alpha
	}
	g1 = 2*(g1-color.val[0])*v.at(3*k+n);
	g2 = 2*(g2-color.val[1])*v.at(3*k+n+1);
	g3 = 2*(g3-color.val[2])*v.at(3*k+n+2);
	g4 = 2*(g4-1);
	return cv::Vec4d(g1,g2,g3,g4);
}

/**
 * Partial derivative of g in respect to u. Note that this function calculates the partial derivatives for all 3 RGB values
 * of u. In the gradient of the minimization function, only the partial derivative of one of those values at a time is needed.
 */
cv::Vec4d dg_u(std::vector<double> &v, int n, cv::Vec3d color, int k){
	double g1 = 0, g2 = 0, g3 = 0, g4 = 0;
	for(size_t i = 0; i < n; i++){
		g1 += v.at(i)*v.at(3*i+n);
		g2 += v.at(i)*v.at(3*i+n+1);
		g3 += v.at(i)*v.at(3*i+n+2);
	}
	g1 = 2*(g1-color.val[0])*v.at(k);
	g2 = 2*(g2-color.val[1])*v.at(k);
	g3 = 2*(g3-color.val[2])*v.at(k);
	return cv::Vec4d(g1,g2,g3,g4);
}

/**
 * Constraint vector g in the refinement step. See Eq. 6 in "Unmixing based soft color segmentation for image manipulation"
 * v: Input vector
 * n: number of colors in color model
 * color: Color of pixel in the input image
 * gt_alpha : the alpha values estimated using filtering step
 */
cv::Vec4d g_refine(std::vector<double> &v, int n, cv::Vec3d color, std::vector<double> gt_alphas){
	double g1 = 0, g2 = 0, g3 = 0, g4 = 0;
	for(size_t i = 0; i < n; i++){
		g1 += v.at(i)*v.at(3*i+n);
		g2 += v.at(i)*v.at(3*i+n+1);
		g3 += v.at(i)*v.at(3*i+n+2);
		g4 += (v.at(i) - gt_alphas.at(i))*(v.at(i) - gt_alphas.at(i));
	}
	return cv::Vec4d(pow(g1-color.val[0],2),pow(g2-color.val[1],2),pow(g3-color.val[2],2),pow(g4,2));
}

/**
 * Partial derivative of g_refine in respect to alpha
 * v: minimization vector
 * n: number of layers
 * color: color of pixel
 * k: index of alpha value for this partial derivative
 * gt_alpha : the alpha values estimated using filtering step
 */
cv::Vec4d dg_refine_a(std::vector<double> &v, int n, cv::Vec3d color, int k, std::vector<double> gt_alphas){
	double g1 = 0, g2 = 0, g3 = 0, g4 = 0;
	for(size_t i = 0; i < n; i++){
		g1 += v.at(i)*v.at(3*i+n);//r
		g2 += v.at(i)*v.at(3*i+n+1);//g
		g3 += v.at(i)*v.at(3*i+n+2);//b
		//g4 += v.at(i);//alpha