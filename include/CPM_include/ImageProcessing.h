
#ifndef _ImageProcessing_h
#define _ImageProcessing_h

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "project.h"
#include <typeinfo>

//----------------------------------------------------------------------------------
// class to handle basic image processing functions
// this is a collection of template functions. These template functions are
// used in other image classes such as BiImage, IntImage and FImage
//----------------------------------------------------------------------------------
enum InterType{ INTER_NN, INTER_LINEAR };

class ImageProcessing
{
public:
	ImageProcessing(void);
	~ImageProcessing(void);
public:

	// basic functions
	template <class T>
	static inline T EnforceRange(const T& x,const int& MaxValue) {return __min(__max(x,0),MaxValue-1);};

	// Values for L are in the range[0, 100] while a and b are roughly in the range[-110, 110].
	template <class T1, class T2>
	static void BGR2Lab(T1* pSrcImage, T2* pDstImage, int width, int height);

	//---------------------------------------------------------------------------------
	// function to interpolate the image plane
	//---------------------------------------------------------------------------------
	template <class T1,class T2>
	static inline void BilinearInterpolate(const T1* pImage,int width,int height,int nChannels,float x,float y,T2* result);

	template <class T1>
	static inline T1 BilinearInterpolate(const T1* pImage,int width,int height,float x,float y);

	// the transpose of bilinear interpolation
	template <class T1,class T2>
	static inline void BilinearInterpolate_transpose(const T1* pImage,int width,int height,int nChannels,float x,float y,T2* result);

	template <class T1>
	static inline T1 BilinearInterpolate_transpose(const T1* pImage,int width,int height,float x,float y);

	template <class T1,class T2>
	static void ResizeImage(const T1* pSrcImage,T2* pDstImage,int SrcWidth,int SrcHeight,int nChannels,float Ratio, InterType type = INTER_LINEAR);

	template <class T1,class T2>
	static void ResizeImage(const T1* pSrcImage, T2* pDstImage, int SrcWidth, int SrcHeight, int nChannels, int DstWidth, int DstHeight, InterType type = INTER_LINEAR);

	//---------------------------------------------------------------------------------
	// functions for 1D filtering
	//---------------------------------------------------------------------------------
	template <class T1,class T2>
	static void hfiltering(const T1* pSrcImage,T2* pDstImage,int width,int height,int nChannels,const float* pfilter1D,int fsize);

	template <class T1,class T2>
	static void vfiltering(const T1* pSrcImage,T2* pDstImage,int width,int height,int nChannels,const float* pfilter1D,int fsize);

	template <class T1,class T2>
	static void hfiltering_transpose(const T1* pSrcImage,T2* pDstImage,int width,int height,int nChannels,const float* pfilter1D,int fsize);

	template <class T1,class T2>
	static void vfiltering_transpose(const T1* pSrcImage,T2* pDstImage,int width,int height,int nChannels,const float* pfilter1D,int fsize);

	//---------------------------------------------------------------------------------
	// functions for 2D filtering
	//---------------------------------------------------------------------------------
	template <class T1,class T2>
	static void filtering(const T1* pSrcImage,T2* pDstImage,int width,int height,int nChannels,const float* pfilter2D,int fsize);

	template <class T1,class T2>
	static void filtering_transpose(const T1* pSrcImage,T2* pDstImage,int width,int height,int nChannels,const float* pfilter2D,int fsize);

	template <class T1,class T2>
	static void Laplacian(const T1* pSrcImage,T2* pDstImage,int width,int height,int nChannels);

	template <class T1, class T2>
	static void Medianfiltering(const T1* pSrcImage, T2* pDstImage, int width, int height, int nChannels, int fsize);

	template <class T1, class T2>
	static void Integral(const T1* pSrcImage, T2* pDstImage, int width, int height, int nChannels);

	template <class T1, class T2> // O(1) time box filtering using cumulative sum
	static void BoxFilter(const T1* pSrcImage, T2* pDstImage, int width, int height, int nChannels, int r, bool norm = true);

	//---------------------------------------------------------------------------------
	// functions for sample a patch from the image
	//---------------------------------------------------------------------------------
	template <class T1,class T2>
	static void getPatch(const T1* pSrcImgae,T2* pPatch,int width,int height,int nChannels,float x,float y,int wsize);

	//---------------------------------------------------------------------------------
	// function to warp image
	//---------------------------------------------------------------------------------
	template <class T1,class T2>
	static void warpImage(T1* pWarpIm2,const T1* pIm1,const T1* pIm2,const T2* pVx,const T2* pVy,int width,int height,int nChannels);

	template <class T1,class T2>
	static void warpImageFlow(T1* pWarpIm2,const T1* pIm1,const T1* pIm2,const T2* pFlow,int width,int height,int nChannels);

	template <class T1,class T2>
	static void warpImage(T1* pWarpIm2,const T1* pIm2,const T2* pVx,const T2* pVy,int width,int height,int nChannels);

	template <class T1,class T2>
	static void warpImage_transpose(T1* pWarpIm2,const T1* pIm2,const T2* pVx,const T2* pVy,int width,int height,int nChannels);

	template <class T1,class T2>
	static void warpImage(T1* pWarpIm2,const T1* pIm2,const T2*flow,int width,int height,int nChannels);

	template <class T1,class T2>
	static void warpImage_transpose(T1* pWarpIm2,const T1* pIm2,const T2* flow,int width,int height,int nChannels);

	template <class T1,class T2,class T3>
	static void warpImage(T1 *pWarpIm2, T3* pMask,const T1 *pIm1, const T1 *pIm2, const T2 *pVx, const T2 *pVy, int width, int height, int nChannels);


	//---------------------------------------------------------------------------------
	// function to crop an image
	//---------------------------------------------------------------------------------
	template <class T1,class T2>
	static void cropImage(const T1* pSrcImage,int SrcWidth,int SrcHeight,int nChannels,T2* pDstImage,int Left,int Top,int DstWidth,int DstHeight);
	//---------------------------------------------------------------------------------

	//---------------------------------------------------------------------------------
	// function to generate a 2D Gaussian
	//---------------------------------------------------------------------------------
	template <class T>
	static void generate2DGaussian(T*& pImage,int wsize,float sigma=-1);

	template <class T>
	static void generate1DGaussian(T*& pImage,int wsize,float sigma=-1);

};

template <class T1, class T2>
void ImageProcessing::BGR2Lab(T1* pSrcImage, T2* pDstImage, int width, int height)
{
	float normFactor = 1.f;
	if (typeid(T1) == typeid(unsigned char)){
		normFactor = 255.f;
	}
	T1* pBGR = pSrcImage;
	T2* pLab = pDstImage;
	const int npix = width*height;

	const float T = 0.008856;
	const float color_attenuation = 1.5f;
	int i;
	for (i = 0; i < npix; i++){
		const float b = pBGR[0] / normFactor;
		const float g = pBGR[1] / normFactor;
		const float r = pBGR[2] / normFactor;
		float X = 0.412453 * r + 0.357580 * g + 0.180423 * b;
		float Y = 0.212671 * r + 0.715160 * g + 0.072169 * b;
		float Z = 0.019334 * r + 0.119193 * g + 0.950227 * b;
		X /= 0.950456;
		Z /= 1.088754;
		float Y3 = pow(Y, 1. / 3);
		float fX = X > T ? pow(X, 1. / 3) : 7.787 * X + 16 / 116.;
		float fY = Y > T ? Y3 : 7.787 * Y + 16 / 116.;
		float fZ = Z > T ? pow(Z, 1. / 3) : 7.787 * Z + 16 / 116.;
		float L = Y > T ? 116 * Y3 - 16.0 : 903.3 * Y;
		float A = 500 * (fX - fY);
		float B = 200 * (fY - fZ);
		// correct L*a*b*: dark area or light area have less reliable colors
		float correct_lab = exp(-color_attenuation*pow(pow(L / 100, 2) - 0.6, 2));
		pLab[0] = L;
		pLab[1] = A*correct_lab;
		pLab[2] = B*correct_lab;
#if 0
		// considering the Values for L are in the range[0, 100] while a and b are roughly in the range[-110, 110],
		// normalize to [0,1]
		pLab[0] /= 220.;
		pLab[1] = (pLab[1] + 110) / 220.;
		pLab[2] = (pLab[2] + 110) / 220.;
#endif