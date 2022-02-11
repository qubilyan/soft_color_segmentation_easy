
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
