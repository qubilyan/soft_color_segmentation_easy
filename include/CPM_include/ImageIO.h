
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