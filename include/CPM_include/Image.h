
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
	void desaturate(Image<T1>& image) const;

	void desaturate();

	template <class T1>
	void collapse(Image<T1>& image,collapse_type type = collapse_average) const;

	void collapse(collapse_type type = collapse_average);

	void flip_horizontal(Image<T>& image);

	void flip_horizontal();

	// function to concatenate images
	template <class T1,class T2>
	void concatenate(Image<T1>& destImage,const Image<T2>& addImage) const;

	template <class T1,class T2>
	void concatenate(Image<T1>& destImage,const Image<T2>& addImage,float ratio) const;

	template <class T1>
	Image<T> concatenate(const Image<T1>& addImage) const;

	// function to separate the channels of the image
	template <class T1,class T2>
	void separate(unsigned firstNChannels,Image<T1>& image1,Image<T2>& image2) const;

	void AddBorder(Image<T>& outImg, int borderWidth) const;

	// function to sample patch
	template <class T1>
	void getPatch(Image<T1>& patch,float x,float y,int fsize) const;

	// function to crop the image
	template <class T1>
	void crop(Image<T1>& patch,int Left,int Top,int Width,int Height) const;

	// basic numerics of images
	template <class T1,class T2>
	void Multiply(const Image<T1>& image1,const Image<T2>& image2);

	template <class T1,class T2>
	void MultiplyAcross(const Image<T1>& image1,const Image<T2>& image2);

	template <class T1,class T2,class T3>
	void Multiply(const Image<T1>& image1,const Image<T2>& image2,const Image<T3>& image3);

	template <class T1>
	void Multiplywith(const Image<T1>& image1);

	template <class T1>
	void MultiplywithAcross(const Image<T1>& image1);

	void Multiplywith(float value);

	template <class T1,class T2>
	void Add(const Image<T1>& image1,const Image<T2>& image2);

	template <class T1,class T2>
	void Add(const Image<T1>& image1,const Image<T2>& image2,float ratio);

	void Add(const T value);

	template <class T1>
	void Add(const Image<T1>& image1,const float ratio);

	template <class T1>
	void Add(const Image<T1>& image1);

	template <class T1,class T2>
	void Subtract(const Image<T1>& image1,const Image<T2>& image2);

	void Subtract(const T value);

	// arithmetic operators
	void square();

	// exp
	void Exp(float sigma = 1);

	// function to normalize an image
	template <class T1>
	void normalize(Image<T1>& image, float minV = -1, float maxV = -1) const;
	void normalize(float minV = -1, float maxV = -1);

	// function to threshold an image
	void threshold(float minV = FLT_MIN, float maxV = FLT_MAX);

	// function to compute the statistics of the image
	float norm2() const;

	float sum() const;

	template <class T1>
	float innerproduct(Image<T1>& image) const;

	template <class T1>
	void Integral(Image<T1>& image) const;

	template <class T1>
	void BoxFilter(Image<T1>& image, int r, bool norm = true) const;

	// function to bilateral smooth flow field
	template <class T1>
	void BilateralFiltering(Image<T1>& other,int fsize,float filter_signa,float range_sigma) const;

	// function to bilateral smooth an image
	//Image<T> BilateralFiltering(int fsize,float filter_sigma,float range_sigma);
	void imBilateralFiltering(Image<T>& result,int fsize,float filter_sigma,float range_sigma) const;

	template <class T1,class T2>
	int kmeansIndex(int pixelIndex,T1& minDistance,const T2* pDictionary,int nVocabulary, int nDim);

	 // convert an image into visual words based on a dictionary
	template <class T1,class T2>
	void ConvertToVisualWords(Image<T1>& result,const T2* pDictionary,int nDim,int nVocabulary);
	
	// get the histogram of an image region
	// the range is [0,imWidth] (x) and [0,imHeight] (y)
	template <class T1>
	Vector<T1> histogramRegion(int nBins,float left,float top,float right,float bottom) const;


	// function for bicubic image interpolation
	template <class T1>
	inline void BicubicCoeff(float a[][4],const T* pIm,const T1* pImDx,const T1* pImDy,const T1* pImDxDy,const int offsets[][2]) const;

	template <class T1,class T2>
	void warpImageBicubic(Image<T>& output,const Image<T1>& imdx,const Image<T1>& imdy, const Image<T1>& imdxdy,const Image<T2>& vx,const Image<T2>& vy) const;

	template <class T1>
	void warpImageBicubic(Image<T>& output,const Image<T1>& vx,const Image<T1>& vy) const;

	template <class T1>
	void warpImageBicubicCoeff(Image<T1>& Coeff) const;

	template <class T1,class T2>
	void warpImageBicubic(Image<T>& output,const Image<T1>& coeff,const Image<T2>& vx,const Image<T2>& vy) const;

	template <class T1,class T2>
	void warpImageBicubicRef(const Image<T>& ref,Image<T>& output,const Image<T1>& imdx,const Image<T1>& imdy, const Image<T1>& imdxdy,const Image<T2>& vx,const Image<T2>& vy) const;

	template <class T1>
	void warpImageBicubicRef(const Image<T>& ref,Image<T>& output,const Image<T1>& vx,const Image<T1>& vy) const;

	template <class T1,class T2>
	void warpImageBicubicRef(const Image<T>& ref,Image<T>& output,const Image<T1>& coeff,const Image<T2>& vx,const Image<T2>& vy) const;

	template <class T1>
	void warpImageBicubicRef(const Image<T>& ref,Image<T>& output,const Image<T1>& flow) const;

	template <class T1>
	void DissembleFlow(Image<T1>& vx,Image<T1>& vy) const;
	// function for image warping
	template <class T1>
	void warpImage(Image<T>& output,const Image<T1>& vx,const Image<T1>& vy) const;

	// function for image warping transpose
	template <class T1>
	void warpImage_transpose(Image<T>& output,const Image<T1>& vx,const Image<T1>& vy) const;

	// function for image warping
	template <class T1>
	void warpImage(Image<T>& output,const Image<T1>& flow) const;

	// function for image warping transpose
	template <class T1>
	void warpImage_transpose(Image<T>& output,const Image<T1>& flow) const;

	// function to get the max
	T maximum() const;

	// function to get min
	T minimum() const;

	void generate2DGuasisan(int winsize,float sigma)
	{
		clear();
		imWidth = imHeight = winsize*2+1;
		nChannels = 1;
		computeDimension();
		ImageProcessing::generate2DGaussian(pData,winsize,sigma);
	}
	void generate1DGaussian(int winsize,float sigma)
	{
		clear();
		imWidth = winsize*2+1;
		imHeight = 1;
		nChannels = 1;
		computeDimension();
		ImageProcessing::generate1DGaussian(pData,winsize,sigma);
	}
	template <class T1>
	void subSampleKernelBy2(Image<T1>& output) const
	{
		int winsize = (imWidth-1)/2;
		int winsize_s  = winsize/2;
		int winlen_s = winsize_s*2+1;
		if(!output.matchDimension(winlen_s,1,1))
			output.allocate(winlen_s,1,1);
		output.pData[winsize_s] = pData[winsize];
		for(int i = 0;i<winsize_s;i++)
		{
			output.pData[winsize_s+1+i] = pData[winsize+2+2*i];
			output.pData[winsize_s-1-i] = pData[winsize-2-2*i];
		}
		output.Multiplywith(1/output.sum());
	}
	void addAWGN(float noiseLevel = 0.05)
	{
		for(int i = 0;i<nElements;i++)
			pData[i] += CStochastic::GaussianSampling()*noiseLevel;
	}

	// file IO
#ifndef _MATLAB
	//bool writeImage(QFile& file) const;
	//bool readImage(QFile& file);
	//bool writeImage(const QString& filename) const;
	//bool readImage(const QString& filename);
#endif

#ifdef _MATLAB
	bool LoadMatlabImage(const mxArray* image,bool IsImageScaleCovnersion=true);
	template <class T1>
	void LoadMatlabImageCore(const mxArray* image,bool IsImageScaleCovnersion=true);

	template <class T1>
	void ConvertFromMatlab(const T1* pMatlabPlane,int _width,int _height,int _nchannels);

	void OutputToMatlab(mxArray*& matrix) const;

	template <class T1>
	void ConvertToMatlab(T1* pMatlabPlane) const;
#endif
};

typedef Image<unsigned char> BiImage;
typedef Image<unsigned char> UCImage;
typedef Image<int> IntImage;
typedef Image<float> FImage;
typedef Image<double> DImage;

//------------------------------------------------------------------------------------------
// constructor
//------------------------------------------------------------------------------------------
template <class T>
Image<T>::Image()
{
	pData=NULL;
	imWidth=imHeight=nChannels=nPixels=nElements=0;
	IsDerivativeImage=false;
	colorType = DATA;
}

//------------------------------------------------------------------------------------------
// constructor with specified dimensions
//------------------------------------------------------------------------------------------
template <class T>
Image<T>::Image(int width,int height,int nchannels)
{
	imWidth=width;
	imHeight=height;
	nChannels=nchannels;
	computeDimension();
	pData=NULL;
	pData = (T*)xmalloc(nElements * sizeof(T));
	if(nElements>0)
		memset(pData,0,sizeof(T)*nElements);
	IsDerivativeImage=false;
	colorType = DATA;
}

template <class T>
Image<T>::Image(const T& value,int _width,int _height,int _nchannels)
{
	pData=NULL;
	allocate(_width,_height,_nchannels);
	setValue(value);
}

#ifndef _MATLAB
//template <class T>
//Image<T>::Image(const QImage& image)
//{
//	pData=NULL;
//	imread(image);
//}
#endif

template <class T>
void Image<T>::allocate(int width,int height,int nchannels)
{
	clear();
	imWidth=width;
	imHeight=height;
	nChannels=nchannels;
	computeDimension();
	pData=NULL;
	colorType = DATA;

	if(nElements>0)
	{
		pData = (T*)xmalloc(nElements * sizeof(T));
		memset(pData,0,sizeof(T)*nElements);
	}
}

template <class T>
template <class T1>
void Image<T>::allocate(const Image<T1> &other)
{
	allocate(other.width(),other.height(),other.nchannels());
	IsDerivativeImage = other.isDerivativeImage();
	colorType = other.colortype();
}

//------------------------------------------------------------------------------------------
// copy constructor
//------------------------------------------------------------------------------------------
template <class T>
Image<T>::Image(const Image<T>& other)
{
	imWidth=imHeight=nChannels=nElements=0;
	pData=NULL;
	copyData(other);
}

//------------------------------------------------------------------------------------------
// destructor
//------------------------------------------------------------------------------------------
template <class T>
Image<T>::~Image()
{
	if (pData != NULL){
		xfree(pData);
	}
}

//------------------------------------------------------------------------------------------
// clear the image
//------------------------------------------------------------------------------------------
template <class T>
void Image<T>::clear()
{
	if (pData != NULL){
		xfree(pData);
	}
	pData=NULL;
	imWidth=imHeight=nChannels=nPixels=nElements=0;
}

//------------------------------------------------------------------------------------------
// reset the image (reset the buffer to zero)
//------------------------------------------------------------------------------------------
template <class T>
void Image<T>::reset()
{
	if(pData!=NULL)
		memset(pData,0,sizeof(T)*nElements);
}

template <class T>
void Image<T>::setValue(const T &value)
{
	for(int i=0;i<nElements;i++)
		pData[i]=value;
}

template <class T>
void Image<T>::setValue(const T& value,int _width,int _height,int _nchannels)
{
	if(imWidth!=_width || imHeight!=_height || nChannels!=_nchannels)
		allocate(_width,_height,_nchannels);
	setValue(value);
}

template <class T>
void Image<T>::setPixel(int row, int col, T* valPtr)
{
	T* ptr = pixPtr(row, col);
	memcpy(ptr, valPtr, sizeof(T)*nChannels);
}

//------------------------------------------------------------------------------------------
// copy from other image
//------------------------------------------------------------------------------------------
template <class T>
void Image<T>::copyData(const Image<T>& other)
{
	imWidth=other.imWidth;
	imHeight=other.imHeight;
	nChannels=other.nChannels;
	nPixels=other.nPixels;
	IsDerivativeImage=other.IsDerivativeImage;
	colorType = other.colorType;

	if(nElements!=other.nElements)
	{
		nElements=other.nElements;	
		if(pData!=NULL)
			xfree(pData);
		pData=NULL;
		pData = (T*)xmalloc(nElements * sizeof(T));
	}
	if(nElements>0)
		memcpy(pData,other.pData,sizeof(T)*nElements);
}

template <class T>
template <class T1>
void Image<T>::copy(const Image<T1>& other)
{
	clear();

	imWidth=other.width();
	imHeight=other.height();
	nChannels=other.nchannels();
	computeDimension();

	IsDerivativeImage=other.isDerivativeImage();
	colorType = other.colortype();

	pData=NULL;
	pData = (T*)xmalloc(nElements * sizeof(T));
	const T1*& srcData=other.data();
	
	// adjust the data according to the data type
	bool isThisFloat = this->IsFloat();
	bool isOtherFloat = other.IsFloat();
	if (isThisFloat != isOtherFloat){
		if (isThisFloat){
			for (int i = 0; i < nElements; i++)
				pData[i] = srcData[i] / 255.0;
		}else{
			for (int i = 0; i < nElements; i++)
				pData[i] = srcData[i] * 255.0;
		}
	}else{
		for (int i = 0; i < nElements; i++)
			pData[i] = srcData[i];
	}
}

template <class T>
void Image<T>::im2float()
{
	if(IsFloat())
		for(int i=0;i<nElements;i++)
			pData[i]/=255;
}

//------------------------------------------------------------------------------------------
// override equal operator
//------------------------------------------------------------------------------------------
template <class T>
Image<T>& Image<T>::operator=(const Image<T>& other)
{
	copyData(other);
	return *this;
}

template <class T>
bool Image<T>::IsFloat() const
{
	if (typeid(T) == typeid(double) || typeid(T) == typeid(float))
		return true;
	else
		return false;
}

template <class T>
template <class T1>
bool Image<T>::matchDimension(const Image<T1>& image) const
{
	if(imWidth==image.width() && imHeight==image.height() && nChannels==image.nchannels())
		return true;
	else
		return false;
}

template <class T>
bool Image<T>::matchDimension(int width, int height, int nchannels) const
{
	if(imWidth==width && imHeight==height && nChannels==nchannels)
		return true;
	else
		return false;
}

//------------------------------------------------------------------------------------------
// function to move this image to a dest image at (x,y) with specified width and height
//------------------------------------------------------------------------------------------
template <class T>
template <class T1>
void Image<T>::moveto(Image<T1>& image,int x0,int y0,int width,int height)
{
	if(width==0)
		width=imWidth;
	if(height==0)
		height=imHeight;
	int NChannels=__min(nChannels,image.nchannels());

	int x,y;
	for(int i=0;i<height;i++)
	{
		y=y0+i;
		if(y>=image.height())
			break;
		for(int j=0;j<width;j++)
		{
			x=x0+j;
			if(x>=image.width())
				break;
			for(int k=0;k<NChannels;k++)
				image.data()[(y*image.width()+x)*image.nchannels()+k]=pData[(i*imWidth+j)*nChannels+k];
		}
	}
}


//------------------------------------------------------------------------------------------
// resize the image
//------------------------------------------------------------------------------------------
template <class T>
bool Image<T>::imresize(float ratio, InterType type/* = INTER_LINEAR*/)
{
	if(pData==NULL)
		return false;

	T* pDstData;
	int DstWidth,DstHeight;
	DstWidth=(float)imWidth*ratio;
	DstHeight=(float)imHeight*ratio;
	pDstData = (T*)xmalloc(DstWidth*DstHeight*nChannels * sizeof(T));

	ImageProcessing::ResizeImage(pData,pDstData,imWidth,imHeight,nChannels,ratio,type);

	xfree(pData);
	pData=pDstData;
	imWidth=DstWidth;
	imHeight=DstHeight;
	computeDimension();
	return true;
}

template <class T>
template <class T1>
void Image<T>::imresize(Image<T1>& result, float ratio, InterType type/* = INTER_LINEAR*/) const
{
	int DstWidth,DstHeight;
	DstWidth=(float)imWidth*ratio;
	DstHeight=(float)imHeight*ratio;
	if(result.width()!=DstWidth || result.height()!=DstHeight || result.nchannels()!=nChannels)
		result.allocate(DstWidth,DstHeight,nChannels);
	else
		result.reset();
	ImageProcessing::ResizeImage(pData,result.data(),imWidth,imHeight,nChannels,ratio,type);
}

template <class T>
template <class T1>
void Image<T>::imresize(Image<T1>& result, int DstWidth, int DstHeight, InterType type/* = INTER_LINEAR*/) const
{
	if(result.width()!=DstWidth || result.height()!=DstHeight || result.nchannels()!=nChannels)
		result.allocate(DstWidth,DstHeight,nChannels);
	else
		result.reset();
	ImageProcessing::ResizeImage(pData,result.data(),imWidth,imHeight,nChannels,DstWidth,DstHeight,type);
}


template <class T>
void Image<T>::imresize(int dstWidth, int dstHeight, InterType type/* = INTER_LINEAR*/)
{
	Image foo(dstWidth,dstHeight,nChannels); // kfj: it should be Image instead of FImage
	ImageProcessing::ResizeImage(pData,foo.data(),imWidth,imHeight,nChannels,dstWidth,dstHeight,type);
	copyData(foo);
}

template <class T>
template <class T1>
void Image<T>::upSampleNN(Image<T1>& output,int ratio) const
{
	int width = imWidth*ratio;
	int height = imHeight*ratio;
	if(!output.matchDimension(width,height,nChannels))
		output.allocate(width,height,nChannels);
	for(int i =  0; i <imHeight; i++)
		for(int j = 0; j<imWidth; j++)
		{
			int offset = (i*imWidth+j)*nChannels;
			for(int ii = 0 ;ii<ratio;ii++)
				for(int jj=0;jj<ratio;jj++)
				{
					int offset1 = ((i*ratio+ii)*width+j*ratio+jj)*nChannels;
					for(int k = 0; k<nChannels; k++)
						output.data()[offset1+k] = pData[offset+k];
				}
		}
}

//------------------------------------------------------------------------------------------
// function of reading or writing images (uncompressed)
//------------------------------------------------------------------------------------------
template <class T>
bool Image<T>::saveImage(const char *filename) const
{
	ofstream myfile(filename,ios::out | ios::binary);
	if(myfile.is_open())
	{
		bool foo = saveImage(myfile);
		myfile.close();
		return foo;
	}
	else
		return false;
}

template <class T>
bool Image<T>::saveImage(ofstream& myfile) const
{
	char type[16];
	sprintf(type,"%s",typeid(T).name());
	myfile.write(type,16);
	myfile.write((char *)&imWidth,sizeof(int));
	myfile.write((char *)&imHeight,sizeof(int));
	myfile.write((char *)&nChannels,sizeof(int));
	myfile.write((char *)&IsDerivativeImage,sizeof(bool));
	myfile.write((char *)pData,sizeof(T)*nElements);
	return true;
}

template <class T>
bool Image<T>::loadImage(const char *filename)
{
	ifstream myfile(filename, ios::in | ios::binary);
	if(myfile.is_open())
	{
		bool foo = loadImage(myfile);
		myfile.close();
		return foo;
	}
	else
		return false;
}

template <class T>
bool Image<T>::loadImage(ifstream& myfile)
{
	char type[16];
	myfile.read(type,16);

	if(strcmpi(type,"uint16")==0)
		sprintf(type,"unsigned short");
	if(strcmpi(type,"uint32")==0)
		sprintf(type,"unsigned int");
	if(strcmpi(type,typeid(T).name())!=0)
	{
		cout<<"The type of the image is different from the type of the object!"<<endl;
		return false;
	}

	int width,height,nchannels;
	myfile.read((char *)&width,sizeof(int));
	myfile.read((char *)&height,sizeof(int));
	myfile.read((char *)&nchannels,sizeof(int));
	if(!matchDimension(width,height,nchannels))
		allocate(width,height,nchannels);
	myfile.read((char *)&IsDerivativeImage,sizeof(bool));
	myfile.read((char *)pData,sizeof(T)*nElements);
	
	return true;
}

//------------------------------------------------------------------------------------------
// function to load the image
//------------------------------------------------------------------------------------------
#ifndef _MATLAB

template <class T>
bool Image<T>::imread(const char* filename)
{
	clear();
	if(ImageIO::loadImage(filename,pData,imWidth,imHeight,nChannels))
	{
		computeDimension();
		colorType = BGR; // when we use qt or opencv to load the image, it's often BGR
		return true;
	}
	return false;
}


//template <class T>
//bool Image<T>::imread(const QString &filename)
//{
//	clear();
//	if(ImageIO::loadImage(filename,pData,imWidth,imHeight,nChannels))
//	{
//		computeDimension();
//		return true;
//	}
//	return false;
//}
//