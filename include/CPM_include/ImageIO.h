
#ifndef _ImageIO_h
#define _ImageIO_h

#include "Util.h"
#include "project.h"
#include "malloc.h"
#include "opencv2/opencv.hpp"
#include <typeinfo>

#ifdef WITH_SSE
#include <xmmintrin.h>
#endif

template <class T>
inline void* xmalloc(T size){
#ifdef WITH_SSE
#ifdef WIN32
	return _aligned_malloc(size, 32);
#else
	return memalign(32, size);
#endif
#else
	return malloc(size);
#endif
}

template <class T>
inline void xfree(T* ptr){
#if defined(WITH_SSE) && defined(WIN32)
	_aligned_free(ptr);
#else
	free(ptr);
#endif
}

#ifdef WITH_SSE

// for windows and linux
typedef union _m128{
	__m128 m;
	__m128i mi;
	float m128_f32[4];
	unsigned short m128i_u16[8];
}hu_m128;

#endif

class ImageIO
{
public:
	enum ImageType{standard, derivative, normalized};
	ImageIO(void);
	~ImageIO(void);
public:
	template <class T>
	static bool loadImage(const char* filename,T*& pImagePlane,int& width,int& height, int& nchannels);
	template <class T>
	static bool saveImage(const char* filename,const T* pImagePlane,int width,int height, int nchannels,ImageType imtype = standard);
	template <class T>
	static void showImage(const char* winname, const T* pImagePlane, int width, int height, int nchannels, ImageType imtype = standard, int waittime = 1);
	template <class T>
	static void showGrayImageAsColor(const char* winname, const unsigned char* pImagePlane, int width, int height, T minV, T maxV, int waittime = 1);
	template <class T>
	static cv::Mat CvmatFromPixels(const T* pImagePlane, int width, int height, int nchannels, ImageType imtype = standard);
	template <class T>
	static void CvmatToPixels(const cv::Mat& cvInImg, T*& pOutImagePlane, int& width, int& height, int& nchannels);
private:
};

template <class T>
bool ImageIO::loadImage(const char *filename, T *&pImagePlane, int &width, int &height, int &nchannels)
{
	cv::Mat im = cv::imread(filename);
	if (im.data == NULL){
		return false;
	}
	pImagePlane = (T*)xmalloc(sizeof(T) * im.total() * im.elemSize());
	CvmatToPixels(im, pImagePlane, width, height, nchannels);
	return true;
}

template <class T>
bool ImageIO::saveImage(const char* filename,const T* pImagePlane,int width,int height, int nchannels,ImageType imtype)
{
	cv::Mat img = CvmatFromPixels(pImagePlane, width, height, nchannels, imtype);
	return cv::imwrite(filename, img);
}

template <class T>
void ImageIO::showImage(const char* winname, const T* pImagePlane, int width, int height, int nchannels, 
	ImageType imtype /*= standard*/, int waittime /*= 1*/)
{
	cv::Mat img = CvmatFromPixels(pImagePlane, width, height, nchannels, imtype);
	cv::imshow(winname, img);
	cv::waitKey(waittime);
}

template <class T>
void ImageIO::showGrayImageAsColor(const char* winname, const unsigned char* pImagePlane, int width, int height, 
	T minV, T maxV, int waittime /*= 1*/)
{
	CColorTable colorTbl;

	// check whether the type is float point
	bool IsFloat = false;
	if (typeid(T) == typeid(double) || typeid(T) == typeid(float) || typeid(T) == typeid(long double))
		IsFloat = true;

	cv::Mat im;
	im.create(height, width, CV_8UC3);
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			unsigned char grayVal = pImagePlane[i*width + j];
			if (IsFloat){
				grayVal = pImagePlane[i*width + j] * 255;
			}
			memcpy(im.data + i*im.step + j * 3, colorTbl[grayVal], 3);
		}
	}

	// show range
	char info[256];
	if (IsFloat)
		sprintf(info, "[%.3f, %.3f]", (float)minV, (float)maxV);
	else
		sprintf(info, "[%d, %d]", (int)minV, (int)maxV);
	cv::putText(im, info, cvPoint(10, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(255, 255, 255));

	//
	cv::imshow(winname, im);
	cv::waitKey(waittime);
}

template <class T>
cv::Mat ImageIO::CvmatFromPixels(const T* pImagePlane, int width, int height, int nchannels, ImageType imtype /*= standard*/)
{
	cv::Mat im;
	switch (nchannels){
	case 1:
		im.create(height, width, CV_8UC1);
		break;
	case 3:
		im.create(height, width, CV_8UC3);
		break;
	case 4:
		im.create(height, width, CV_8UC4);
		break;
	default:
		return im;
	}
	// check whether the type is float point
	bool IsFloat = false;
	if (typeid(T) == typeid(double) || typeid(T) == typeid(float) || typeid(T) == typeid(long double))
		IsFloat = true;

	double Max, Min;
	int nElements = width*height*nchannels;
	switch (imtype){
	case standard:
		break;
	case derivative:
		// find the max of the absolute value
		Max = pImagePlane[0];
		for (int i = 0; i < nElements; i++)
			Max = __max(Max, fabs((double)pImagePlane[i]));
		Min = -Max;
		break;
	case normalized:
		Max = Min = pImagePlane[0];
		for (int i = 0; i < nElements; i++)
		{
			Max = __max(Max, pImagePlane[i]);
			Min = __min(Min, pImagePlane[i]);
		}
		break;
	}
	if (typeid(T) == typeid(unsigned char) && imtype == standard)
	{
		for (int i = 0; i < height; i++)
			memcpy(im.data + i*im.step, pImagePlane + i*im.step, width*nchannels);
	}
	else
	{
		for (int i = 0; i < height; i++)
		{
			int offset1 = i*width*nchannels;
			int offset2 = i*im.step;
			for (int j = 0; j < im.step; j++)