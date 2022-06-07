
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
	}
	g1 = 2*(g1-color.val[0])*v.at(3*k+n);
	g2 = 2*(g2-color.val[1])*v.at(3*k+n+1);
	g3 = 2*(g3-color.val[2])*v.at(3*k+n+2);
	g4 = 2*( v.at(k) - gt_alphas.at(k)); 
	return cv::Vec4d(g1,g2,g3,g4);
}



/**
 * Partial derivative of the Mahalanobis distance in respect to u
 */
double d_maha(cv::Vec3d mean, cv::Matx33d cov, cv::Vec3d u, int i){ //the square has been removed from the mahalanobis distance so we don't need to deal with sqrt in derivative
    cv::Matx33d cov_inv = cov.inv();
	cv::Vec3d df_m;
	double df_i;
	df_m = cv::Vec3d(2*cov_inv*(cv::Mat(u - mean)));
	df_i = df_m.val[i];
    //return 2*(u.val[i]-mean.val[i])*(cov_inv.col(i).val[0]+cov_inv.col(i).val[1]+cov_inv.col(i).val[2]);//mg checked, 
	return df_i;
	}

/**
 * Minimization function for the unmixing step. See Line 1 in Algorithm 1 in
 * "Interactive High-Quality Green-Screen Keying via Color Unmixing"
 */
double min_f(std::vector<double> &v, void* params){
	InputParams* param = static_cast<InputParams*>(params);
	std::vector<cv::Vec3d> means = std::get<0>((*param));
	std::vector<cv::Matx33d> covs = std::get<1>((*param));
	cv::Vec4d lambda = std::get<2>((*param));
	cv::Vec3d color = std::get<3>((*param));
	double p = std::get<4>((*param));
	bool s =std:: get<5>((*param));
	cv::Vec4d g_vec = g(v, means.size(), color); //constraint vector 
	return energy(v, means, covs, s) + g_vec.dot(lambda) + 0.5*p*pow(cv::norm(g_vec, cv::NORM_L2),2);
}

/**
 * Gradient of the minimization function for the unmixing step. Returns a vector where each element is the partial
 * derivative of input vector v in respect to that element (#Jo: rather of energy(v) in repect to that element)
 */
std::vector<double> min_df(std::vector<double> &v, void* params){
	InputParams* param = static_cast<InputParams*>(params);
	std::vector<cv::Vec3d> means = std::get<0>((*param));
	std::vector<cv::Matx33d> covs = std::get<1>((*param));
	cv::Vec4d lambda = std::get<2>((*param));
	cv::Vec3d color = std::get<3>((*param));
	double p = std::get<4>((*param));
	bool s = std::get<5>((*param));
	int n = means.size();
	std::vector<double> df(4*n);
	double dist, sparse, alpha, dot_prod, g_norm, u1, u2, u3, grad_a, grad_u1, grad_u2, grad_u3;
	double sum_alpha = 0.0, sum_squared = 0.0, sum_u1 = 0.0, sum_u2 = 0.0, sum_u3 = 0.0;
	cv::Vec3d u;
	for(size_t i = 0; i < n; i++){
		alpha = v.at(i);
		u1 = v.at(3*i+n);
		u2 = v.at(3*i+n+1);
		u3 = v.at(3*i+n+2);
		sum_alpha += alpha;
		sum_squared += pow(alpha,2);
		sum_u1 += alpha*u1; //B_i * alpha_i
		sum_u2 += alpha*u2; //G_i * alpha_i
		sum_u3 += alpha*u3; //R_i * alpha_i
	}
	for(size_t i = 0; i < n; i++){
		alpha = v.at(i);
		u1 = v.at(3*i+n);
		u2 = v.at(3*i+n+1);
		u3 = v.at(3*i+n+2);
		u = cv::Vec3d(u1,u2,u3);
		//Alphas
		dist = pow(cv::Mahalanobis(u, means.at(i), (covs.at(i)).inv()),2);
		if(s){
			if(sum_squared < 0.0000000001){
				sparse = 0;
			}else{
				sparse = (constants::sigma*sum_squared-(constants::sigma*sum_alpha*2*alpha))/(pow(sum_squared,2));
			}
		} else {
			sparse = 0;
		}
		dot_prod = lambda.dot(dg_a(v, means.size(), color, i));
		g_norm = 0.5*p* (4*pow(sum_u1-color.val[0],3)*u1
					+ 4*pow(sum_u2-color.val[1],3)*u2 
					+ 4*pow(sum_u3-color.val[2],3)*u3
					+ 4*pow(sum_alpha-1,3));
		grad_a = dist+sparse+dot_prod+g_norm;
		df.at(i) = grad_a;
		//Colors
		//u1
		dist = alpha*d_maha(means.at(i),covs.at(i),u,0);
		dot_prod = lambda.dot(cv::Vec4d(dg_u(v, means.size(), color, i).val[0],0,0,0));
		g_norm = 0.5*p*4*pow(sum_u1-color.val[0],3)*alpha;
		grad_u1 = dist+dot_prod+g_norm;
		df.at(3*i+n) = grad_u1;
		//u2
		dist = alpha*d_maha(means.at(i),covs.at(i),u,1);
		dot_prod = lambda.dot(cv::Vec4d(0,dg_u(v, means.size(), color, i).val[1],0,0));
		g_norm = 0.5*p*4*pow(sum_u2-color.val[1],3)*alpha;
		grad_u2 = dist+dot_prod+g_norm;
		df.at(3*i+n+1) = grad_u2;
		//u3
		dist = alpha*d_maha(means.at(i),covs.at(i),u,2);
		dot_prod = lambda.dot(cv::Vec4d(0,0,dg_u(v, means.size(), color, i).val[2],0));
		g_norm = 0.5*p*4*pow(sum_u3-color.val[2],3)*alpha;
		grad_u3 = dist+dot_prod+g_norm;
		df.at(3*i+n+2) = grad_u3;
	}
	// Apply box constraints
	for(size_t i = 0; i < 4*n; i++){
		if((df.at(i) < 0 && v.at(i) >= 1) || (df.at(i) > 0 && v.at(i) <= 0)){
			df.at(i) = 0;
		}
	}
	return df;
}



/**
 * mg - Minimization function for the color refinement step. See equation (4) and(6) of unmixing paper
 * 
 */
double min_refine_f(std::vector<double> &v, void* params){
	InputParams* param = static_cast<InputParams*>(params);
	std::vector<cv::Vec3d> means = std::get<0>((*param));
	std::vector<cv::Matx33d> covs = std::get<1>((*param));
	cv::Vec4d lambda = std::get<2>((*param));
	cv::Vec3d color = std::get<3>((*param));
	double p = std::get<4>((*param));
	bool s = 0;
	std::vector<double> gt_alpha = std::get<7>((*param));
	cv::Vec4d g_vec = g_refine(v, means.size(), color, gt_alpha); //constraint vector
	return energy(v, means, covs, s) + g_vec.dot(lambda) + 0.5*p*pow(cv::norm(g_vec, cv::NORM_L2),2);
}

/**
 * Gradient of the minimization function for the color refinement step. Returns a vector where each element is the partial
 * derivative of input vector v in respect to that element.
 */
std::vector<double> min_refine_df(std::vector<double> &v, void* params){
	InputParams* param = static_cast<InputParams*>(params);
	std::vector<cv::Vec3d> means = std::get<0>((*param));
	std::vector<cv::Matx33d> covs = std::get<1>((*param));
	cv::Vec4d lambda = std::get<2>((*param));
	cv::Vec3d color = std::get<3>((*param));
	double p = std::get<4>((*param));
	bool s = std::get<5>((*param));
	std::vector<double> gt_alpha = std::get<7>((*param));
	int n = means.size();
	std::vector<double> df(4*n);
	double dist, sparse, alpha, alphai_gt, dot_prod, g_norm, u1, u2, u3, grad_a, grad_u1, grad_u2, grad_u3;
	double diff_gt_alpha = 0.0, sum_squared = 0.0, sum_u1 = 0.0, sum_u2 = 0.0, sum_u3 = 0.0;
	cv::Vec3d u;
	for(size_t i = 0; i < n; i++){
		alpha = v.at(i);
		alphai_gt = gt_alpha.at(i);
		u1 = v.at(3*i+n);
		u2 = v.at(3*i+n+1);
		u3 = v.at(3*i+n+2);
		diff_gt_alpha += pow((alpha - alphai_gt),2);
		sum_squared += pow(alpha,2);
		sum_u1 += alpha*u1; //B_i * alpha_i
		sum_u2 += alpha*u2; //G_i * alpha_i
		sum_u3 += alpha*u3; //R_i * alpha_i
	}
	for(size_t i = 0; i < n; i++){
		alpha = v.at(i);
		alphai_gt = gt_alpha.at(i);
		u1 = v.at(3*i+n);
		u2 = v.at(3*i+n+1);
		u3 = v.at(3*i+n+2);
		u = cv::Vec3d(u1,u2,u3);
		//Alphas - mg changed this so that the alpha values are not estimated in the final step - the filtered alphas are used instead - if they change, they may not add to 1 after optimisation.
		//these are the rows that were removed:
		/*************************
		//dist = pow(cv::Mahalanobis(u, means.at(i), (covs.at(i)).inv()),2);
		//sparse = 0;
		//dot_prod = lambda.dot(dg_refine_a(v, means.size(), color, i, gt_alpha));
		//g_norm = 0.5*p* (4*pow(sum_u1-color.val[0],3)*u1
		//			+ 4*pow(sum_u2-color.val[1],3)*u2 
		//			+ 4*pow(sum_u3-color.val[2],3)*u3
		//			+ 4*(diff_gt_alpha)*(alpha - alphai_gt)); //mg checked
		//grad_a = dist+sparse+dot_prod+g_norm;
		//df.at(i) = grad_a;
		/****************************/
		df.at(i) = 0;
		//Colors
		//u1
		dist = alpha*d_maha(means.at(i),covs.at(i),u,0);
		dot_prod = lambda.dot(cv::Vec4d(dg_u(v, means.size(), color, i).val[0],0,0,0));
		g_norm = 0.5*p*4*pow(sum_u1-color.val[0],3)*alpha;
		grad_u1 = dist+dot_prod+g_norm;
		df.at(3*i+n) = grad_u1;
		//u2
		dist = alpha*d_maha(means.at(i),covs.at(i),u,1);
		dot_prod = lambda.dot(cv::Vec4d(0,dg_u(v, means.size(), color, i).val[1],0,0));
		g_norm = 0.5*p*4*pow(sum_u2-color.val[1],3)*alpha;
		grad_u2 = dist+dot_prod+g_norm;
		df.at(3*i+n+1) = grad_u2;
		//u3
		dist = alpha*d_maha(means.at(i),covs.at(i),u,2);
		dot_prod = lambda.dot(cv::Vec4d(0,0,dg_u(v, means.size(), color, i).val[2],0));
		g_norm = 0.5*p*4*pow(sum_u3-color.val[2],3)*alpha;
		grad_u3 = dist+dot_prod+g_norm;
		df.at(3*i+n+2) = grad_u3;
	}
	// Apply box constraints
	for(size_t i = 0; i < 4*n; i++){
		if((df.at(i) < 0 && v.at(i) >= 1) || (df.at(i) > 0 && v.at(i) <= 0)){
			df.at(i) = 0;
		}
	}
	//mg added this to make sure that the alpha values don't change in the final refinement step - can rejig this code in future to make better
	//for(size_t i = 0; i < n; i++){
	//	df.at(i) = 0;
	//}
	return df;
}




/**
 * Returns -v for input vector v
 */
std::vector<double> negate_v(std::vector<double> v){
	std::vector<double> r(v.size());
	for(unsigned int i = 0; i < v.size(); i++){
		r.at(i) = -v.at(i);
	}
	return r;
}

/**
 * The beta value for the polak-ribiere conjugate gradient descent
 */
double b_pr(std::vector<double> &a, std::vector<double> &b){
	double r1 = 0.0, r2 = 0.0;
	for(unsigned int i = 0; i < a.size(); i++){
		r1 += a.at(i)*(a.at(i)-b.at(i));
		r2 += b.at(i)*b.at(i);
	}
	if(r2 != 0){
		return r1/r2;
	} else {
		return 0;
	}
}

/**
 * Make Input vector v a unit vector
 */
std::vector<double> make_unit(std::vector<double> v){
	double l = 0.0;
	std::vector<double> r(v.size());
	for(double d : v){
		l += pow(d,2);
	}
	double k = sqrt(l);
	for(unsigned int i = 0; i < v.size(); i++){
		if(k != 0){
			r.at(i) = v.at(i)/k;
		}
	}
	return r;
}

/**
 * Clip a double value into range [0,1]
 */
double clip(double x){
	if(x > 1){
		return 1;
	} else if(x < 0){
		return 0;
	} else {
		return x;
	}
}

/**
 * Clip a vector into range [0,1]
 */
std::vector<double> clip(std::vector<double> &v){
	for(unsigned int i = 0; i < v.size(); i++){
		v.at(i) = (v.at(i)>1.0)?1.0:v.at(i);
		v.at(i) = (v.at(i)<0.0)?0.0:v.at(i);
	}
	return v;
}

/**
 * Linear transformation of two vectors and a scalar: a+c*b
 */
std::vector<double> lin_transform(std::vector<double> &a, std::vector<double> &b, double c){
	std::vector<double> r(a.size());
	for(unsigned int i = 0; i < a.size(); i++){
		r.at(i) = a.at(i)+c*b.at(i);
	}
	return r;
}

/**
 * Backtracking line-search
 * Start direction x, search direction d
 */
double line_search(vFunctionCall f, std::vector<double> &x, std::vector<double> d, void* params)
{

	double in = f(x,params);
	double a_k = .5;