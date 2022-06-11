
#include <opencv2/opencv.hpp>
#include "Pixel.h"
#include "Unmixing.h"
#include "ThreadPool.h"
#include <chrono>
#include <vector>
#include <string>
#include "guidedfilter.h"
#include "ColorModel.h"
#include <sys/stat.h> 


int makeDirectories(std::string result_folder_name)
{
    //create folder to save output layers and videos cleanly
    int status;
    status = mkdir(result_folder_name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if( status != 0){
		std::cout << "Error mkdir: " <<strerror(errno)<<std::endl;
	    return EXIT_FAILURE;
	}
    result_folder_name += "/";
	status = mkdir((result_folder_name +"output_layers").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if( status != 0){
		std::cout << "Error mkdir: " <<strerror(errno)<<std::endl;
	    return EXIT_FAILURE;
	}
	status = mkdir((result_folder_name +"sum_frames").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if( status != 0){
		std::cout << "Error mkdir: " <<strerror(errno)<<std::endl;
	    return EXIT_FAILURE;
	}
    return EXIT_SUCCESS;
}

void saveLayers(std::string dir, std::vector<cv::Mat> layers, std::string step)
{
    for (int l = 0; l<layers.size(); l++)
        {
            std::string l_str = std::string(2 - std::to_string(l).length(), '0') + std::to_string(l);
            imwrite(dir + "/output_layers/" + step + l_str + ".png",
                layers[l]);
        }
}

void sumLayers(cv::Mat sum, std::vector<cv::Mat> layers)
{
	int n = layers.size();

	//for every pixel
	for(size_t i = 0; i < sum.rows; i++)
	{
		for(size_t j = 0; j < sum.cols; j++)
		{
			double alpha_unit;
			double sum_u1 = 0.0, sum_u2 = 0.0, sum_u3 = 0.0, sum_alpha = 0.0;
			cv::Vec4b& pix = sum.at<cv::Vec4b>(i,j);
			//for every layer
			for(size_t l = 0; l < n; l++){
				cv::Mat layer = layers.at(l);
				cv::Vec4b v = layer.at<cv::Vec4b>(i,j);

				alpha_unit = v[3] / 255.0;
				sum_u1 += alpha_unit*v[0]; //B_i * alpha_i
				sum_u2 += alpha_unit*v[1]; //G_i * alpha_i
				sum_u3 += alpha_unit*v[2]; //R_i * alpha_i
				sum_alpha += v[3];
			}

			// clip because 256 becomes 1 in uchar
			pix[0] = (sum_u1>255)?255:sum_u1;
			pix[1] = (sum_u2>255)?255:sum_u2;
			pix[2] = (sum_u3>255)?255:sum_u3;
			pix[3] = (sum_alpha>255)?255:sum_alpha;
		}
	}
}



int main( int argc, char** argv )
{
	double tau;
	if(argc == 1){
		std::cout << "Usage: ./SoftSegmentation <path/to/image/> <tau_parameterer(optional)> <output_dir>" << std::endl; 
		return -1;
	}else if(argc == 2){
		tau = constants::tau;
	}else {
		tau = std::stod(argv[2]);
	}

    std::vector<cv::Vec3d> means;
    std::vector<cv::Matx33d> covs;

    // Load image
	char* imageName = argv[1];
    char* output_dir_name = argv[3];
	std::string imageName_short(imageName);//mg_added
  	std::string imageName_s(imageName_short.begin()+3, imageName_short.end()-4);
	std::string dir_name(output_dir_name);

	//create directories that we'll save images into
	if(makeDirectories(std::string("../") + dir_name) != 0){
        std::cout <<"Error making directories"<<std::endl;
        return false;
    }

	  
	//Read input image and write it to output directory
	cv::Mat image;
	image = cv::imread(imageName, 1);
	cv::imwrite(std::string("../") + dir_name + "/input_image.png", image); //save input image for reference
	if( !image.data ){
	   printf( " No image data \n " );
	   return -1;
	}


	/*************************************************************
	 *            To Begin: Compute Global Color Model             *
	 *************************************************************/
	//create terminal output
	std::cout << "" << std::endl;
	std::cout << "Colour Model Estimation Step..." << std::endl;
	std::cout << "" << std::endl;


    auto t_start_CM = std::chrono::high_resolution_clock::now();
