
#pragma once

#include "project.h"
#include <stdio.h>
#include <string.h>
#include <memory.h>
#include "ImageProcessing.h"
#include <iostream>
#include <fstream>
#include <typeinfo>
#include "Vector.h"
#include "Stochastic.h"

#ifndef _MATLAB
	#include "ImageIO.h"
#else
	#include "mex.h"
#endif

using namespace std;

enum collapse_type{collapse_average,collapse_max,collapse_min};
enum color_type{ DATA, GRAY, RGB, BGR, LAB };

// template class for image
template <class T>
class Image
{
public:
	T* pData;
protected:
	int imWidth,imHeight,nChannels;
	int nPixels,nElements;
	bool IsDerivativeImage;
	color_type colorType;
public:
	Image(void);
	Image(int width,int height,int nchannels=1);
	Image(const T& value,int _width,int _height,int _nchannels=1);
	Image(const Image<T>& other);
	~Image(void);
	virtual Image<T>& operator=(const Image<T>& other);

	virtual inline void computeDimension(){nPixels=imWidth*imHeight;nElements=nPixels*nChannels;};

	virtual void allocate(int width,int height,int nchannels=1);
	
	template <class T1>
	void allocate(const Image<T1>& other);

	virtual void clear();
	virtual void reset();
	virtual void copyData(const Image<T>& other);
	void setValue(const T& value);
	void setValue(const T& value,int _width,int _height,int _nchannels=1);
	void setPixel(int row, int col, T* valPtr);

	T immax() const
	{
		T Max=pData[0];
		for(int i=1;i<nElements;i++)
			Max=__max(Max,pData[i]);
		return Max;
	};
	T immin() const{
		T Min=pData[0];
		for(int i=1;i<nElements;i++)
			Min=__min(Min,pData[i]);
		return Min;
	}
	template <class T1>
	void copy(const Image<T1>& other);

	void im2float();

	// function to access the member variables
	inline const T& operator [] (int index) const {return pData[index];};
	inline T& operator[](int index) {return pData[index];};

	inline T* rowPtr(int row){ return pData + row*imWidth*nChannels; };
	inline T* pixPtr(int row, int col){ return pData + (row*imWidth + col)*nChannels; };

	inline T*& data(){return pData;};
	inline const T*& data() const{return (const T*&)pData;};
	inline int width() const {return imWidth;};
	inline int cols() const {return imWidth;};
	inline int height() const {return imHeight;};
	inline int rows() const {return imHeight;};
	inline int nchannels() const {return nChannels;};
	inline int npixels() const {return nPixels;};
	inline int nelements() const {return nElements;};
	inline bool isDerivativeImage() const {return IsDerivativeImage;};
	inline color_type colortype() const{return colorType;};

	bool IsFloat () const;
	bool IsEmpty() const {if(nElements==0) return true;else return false;};
	bool IsInImage(int x,int y) const {if(x>=0 && x<imWidth && y>=0 && y<imHeight) return true; else return false;};

	template <class T1>
	bool matchDimension  (const Image<T1>& image) const;

	bool matchDimension (int width,int height,int nchannels) const;

	inline void setDerivative(bool isDerivativeImage=true){IsDerivativeImage=isDerivativeImage;};

	bool BoundaryCheck() const;
	// function to move this image to another one
	template <class T1>
	void moveto(Image<T1>& image,int x,int y,int width=0,int height=0);

	// function of basic image operations
	virtual bool imresize(float ratio, InterType type = INTER_LINEAR);
	template <class T1>
	void imresize(Image<T1>& result, float ratio, InterType type = INTER_LINEAR) const;
	void imresize(int dstWidth, int dstHeight, InterType type = INTER_LINEAR);
	template <class T1>
	void imresize(Image<T1>& result, int dstWidth, int dstHeight, InterType type = INTER_LINEAR) const;

	template <class T1>
	void upSampleNN(Image<T1>& result,int ratio) const;

	// image IO's
	virtual bool saveImage(const char* filename) const;
	virtual bool loadImage(const char* filename);
	virtual bool saveImage(ofstream& myfile) const;
	virtual bool loadImage(ifstream& myfile);
#ifndef _MATLAB
	virtual bool imread(const char* filename);
	virtual bool imwrite(const char* filename) const;
	virtual bool imwrite(const char* filename,ImageIO::ImageType) const;
	virtual void imshow(char* winname, int waittime = 1) const;
	virtual void imagesc(char* winname, int waittime = 1) const;

	//virtual bool imread(const QString& filename);
	//virtual void imread(const QImage& image);

	//virtual bool imwrite(const QString& filename,int quality=100) const;
	//virtual bool imwrite(const QString& filename,ImageIO::ImageType imagetype,int quality=100) const;
	//virtual bool imwrite(const QString& fileanme,T min,T max,int quality=100) const;
#else
	virtual bool imread(const char* filename) const {return true;};
	virtual bool imwrite(const char* filename) const {return true;};
#endif

	// matches stored as x1 y1 x2 y2
	template <class T1>
	void DrawLines(T1* posPtr, int lineCnt, float b, float g, float r, int thickness = 1);

	// change the color space to CIELab
	void ToLab() const;

	template <class T1>
	void ToLab(Image<T1>& image) const;

	template <class T1>
	Image<T1> dx (bool IsAdvancedFilter=false) const;

	template <class T1>
	void dx(Image<T1>& image,bool IsAdvancedFilter=false) const;

	template<class T1>
	Image<T1> dy(bool IsAdvancedFilter=false) const;

	template <class T1>
	void dy(Image<T1>& image,bool IsAdvancedFilter=false) const;

	template <class T1>
	void dxx(Image<T1>& image) const;

	template <class T1>
	void dyy(Image<T1>& image) const;

	template <class T1>
	void dxy(Image<T1>& image) const;

	template <class T1>
	void laplacian(Image<T1>& image) const;

	template <class T1>
	void gradientmag(Image<T1>& image) const;

	void GaussianSmoothing(float sigma,int fsize);

	template <class T1>
	void GaussianSmoothing(Image<T1>& image,float sigma,int fsize) const;

	template <class T1>
	void GaussianSmoothing_transpose(Image<T1>& image,float sigma,int fsize) const;

	template <class T1>
	void smoothing(Image<T1>& image,float factor=4);

	template <class T1>
	Image<T1> smoothing(float factor=4);

	void smoothing(float factor=4);

	void MedianFiltering(int fsize = 2);

	// function for filtering
	template <class T1>
	void imfilter(Image<T1>& image,const float* filter,int fsize) const;

	template <class T1,class T2>
	void imfilter(Image<T1>& image,const Image<T2>& kernel) const;

	template <class T1>
	Image<T1> imfilter(const float* filter,int fsize) const;

	template <class T1>
	void imfilter_h(Image<T1>& image,float* filter,int fsize) const;

	template <class T1>
	void imfilter_v(Image<T1>& image,float* filter,int fsize) const;

	template <class T1>
	void imfilter_hv(Image<T1>& image,const float* hfilter,int hfsize,const float* vfilter,int vfsize) const;

	template<class T1>
	void imfilter_hv(Image<T1>& image,const Image<float>& hfilter,const Image<float>& vfilter) const;

	// function for filtering transpose
	template <class T1>
	void imfilter_transpose(Image<T1>& image,const float* filter,int fsize) const;

	template <class T1,class T2>
	void imfilter_transpose(Image<T1>& image,const Image<T2>& kernel) const;

	template <class T1>
	Image<T1> imfilter_transpose(const float* filter,int fsize) const;

	template <class T1>
	void imfilter_h_transpose(Image<T1>& image,float* filter,int fsize) const;

	template <class T1>
	void imfilter_v_transpose(Image<T1>& image,float* filter,int fsize) const;

	template <class T1>
	void imfilter_hv_transpose(Image<T1>& image,const float* hfilter,int hfsize,const float* vfilter,int vfsize) const;

	template<class T1>
	void imfilter_hv_transpose(Image<T1>& image,const Image<float>& hfilter,const Image<float>& vfilter) const;

	// function to desaturating
	template <class T1>