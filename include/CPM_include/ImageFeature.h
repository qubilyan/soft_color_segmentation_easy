
#pragma once
#include "Image.h"
#include "math.h"
#include "memory.h"
#include <vector>

#ifndef M_PI
#define M_PI 3.1415926535897932384626433
#endif


class ImageFeature
{
public:
	ImageFeature(void);
	~ImageFeature(void);

	template <class T>
	static void imSIFT(const Image<T>& imsrc, UCImage& imsift, int cellSize = 2, int stepSize = 1, bool IsBoundaryIncluded = false, int nBins = 8);

	template <class T>
	static void imSIFT(const Image<T>& imsrc, UCImage& imsift, const vector<int> cellSizeVect, int stepSize = 1, bool IsBoundaryIncluded = false, int nBins = 8);

};

template <class T>
void ImageFeature::imSIFT(const Image<T>& imsrc, UCImage &imsift, int cellSize, int stepSize, bool IsBoundaryIncluded, int nBins)
{
	if(cellSize<=0)
	{
		cout<<"The cell size must be positive!"<<endl;
		return;
	}

	// this parameter controls decay of the gradient energy falls into a bin
	// run SIFT_weightFunc.m to see why alpha = 9 is the best value
	int alpha = 9;

	int width = imsrc.width(), height = imsrc.height(), nchannels = imsrc.nchannels();
	int nPixels = width*height;
	FImage imdx(width, height, nchannels), imdy(width, height, nchannels);
	// compute the derivatives;
#if 0
	imsrc.dx(imdx, true);
	imsrc.dy(imdy, true);
#else
	// sobel, more robust
	float xKernel[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	float yKernel[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
	ImageProcessing::filtering(imsrc.pData, imdx.pData, width, height, nchannels, xKernel, 1);
	ImageProcessing::filtering(imsrc.pData, imdy.pData, width, height, nchannels, yKernel, 1);
#endif

	// get the maximum gradient over the channels and estimate the normalized gradient
	FImage magsrc(width,height,nchannels),mag(width,height),gradient(width,height,2);
	float Max;
	for(int i=0;i<nPixels;i++)
	{
		int offset = i*nchannels;
		for(int j = 0;j<nchannels;j++)
			magsrc.pData[offset+j] = sqrt(imdx.pData[offset+j]*imdx.pData[offset+j]+imdy.pData[offset+j]*imdy.pData[offset+j]);
		Max = magsrc.pData[offset];
		if(Max!=0)
		{
			gradient.pData[i*2] = imdx.pData[offset]/Max;
			gradient.pData[i*2+1] = imdy.pData[offset]/Max;
		}
		for(int j = 1;j<nchannels;j++)
		{
			if(magsrc.pData[offset+j]>Max)
			{
				Max = magsrc.pData[offset+j];
				gradient.pData[i*2] = imdx.pData[offset+j]/Max;
				gradient.pData[i*2+1] = imdy.pData[offset+j]/Max;
			}
		}
		mag.pData[i] = Max;
	}

	// get the pixel-wise energy for each orientation band
	FImage imband(width,height,nBins);
	float theta = M_PI*2/nBins;
	float _cos,_sin,temp;
	for(int k = 0;k<nBins;k++)
	{
		_sin    = sin(theta*k);
		_cos   = cos(theta*k);
		for(int i = 0;i<nPixels; i++)
		{
			temp = __max(gradient.pData[i*2]*_cos + gradient.pData[i*2+1]*_sin,0);
			if(alpha>1)