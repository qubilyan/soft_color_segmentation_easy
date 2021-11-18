/*

Code of the Coarse-to-Fine PatchMatch, published at CVPR 2016 in
"Efficient Coarse-to-Fine PatchMatch for Large Displacement Optical Flow"
by Yinlin.Hu, Rui Song and Yunsong Li.

Email: huyinlin@gmail.com

Version 1.2

Copyright (C) 2016 Yinlin.Hu

Usages:

The program "cpm.exe" has been built and tested on Windows 7.

USAGE: cpm.exe img1Name img2Name outMatchName

Explanations:

The output of the program is a text file, which is in the format of "x1,y1,x2,y2"
corresponding to one match per line.

*/

#ifndef _CPM_H_
#define _CPM_H_

#include "ImagePyramid.h"

class CPM
{
public:
	CPM();
	~CPM();

	int Matching(FImage& img1, FImage& img2, FImage& outMatches);
	void SetStereoFlag(int needStereo);
	void SetStep(int step);

private:
	void imDaisy(FImage& img, UCImage& outFtImg);
	void CrossCheck(IntImage& seeds, FImage& seedsFlow, FImage& seedsFlow2, IntImage& kLabel2, int* valid, float th);
	float MatchCost(FImage& img1, FImage& img2, UCImag