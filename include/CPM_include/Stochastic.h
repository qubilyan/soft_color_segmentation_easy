
#ifndef STOCHASTIC_H
#define STOCHASTIC_H

#include "math.h"
#include "stdlib.h"
#include "project.h"
#include "memory.h"

#define _Release_2DArray(X,i,length) for(i=0;i<length;i++) if(X[i]!=NULL) delete X[i]; delete []X

template <typename T>
T _abs(T x)
{
	return x >= 0 ? x:-x;
}

#ifndef PI
#define PI 3.1415927
#endif

enum SortType{SortAscending,SortDescending};

class CStochastic
{
public:
	CStochastic(void);
	~CStochastic(void);
	static void ConvertInt2String(int x,char* string,int BitNumber=3);
	static double UniformSampling();
	static int UniformSampling(int R);
	static double GaussianSampling();
	template <class T> static void GetMeanVar(T* signal,int length,double* mean,double* var);
	static int Sampling(double* Density,int NumSamples);
	static double GetMean(double *signal,int length);
	static void Generate1DGaussian(double* pGaussian,int size,double sigma=0);
	static void Generate2DGaussian(double* pGaussian,int size,double sigma=0);
	static double entropy(double* pDensity,int n);

	template <class T> static T sum(int NumData,T* pData);
	template <class T> static void Normalize(int NumData,T* pData);
	template <class T> static T mean(int NumData, T* pData);
	template <class T> static void sort(int number, T* pData,int *pIndex,SortType m_SortType=SortDescending);
	template <class T> static T Min(int NumData, T* pData);
	template <class T> static T Min(int NumData, T* pData1,T* pData2);
	template <class T> static T Max(int NumData ,T* pData);
	template <class T> static int FindMax(int NumData,T* pData);
	template <class T1,class T2> static void ComputeVectorMean(int Dim,int NumData,T1* pData,T2* pMean,double* pWeight=NULL);
	template <class T1,class T2> static void ComputeMeanCovariance(int Dim,int NumData,T1* pData,T2* pMean,T2* pCovarance,double* pWeight=NULL);
	template <class T1,class T2> static double VectorSquareDistance(int Dim,T1* pVector1,T2* pVector2);
	template <class T1> static void KMeanClustering(int Dim,int NumData,int NumClusters,T1* pData,int *pPartition,double** pClusterMean=NULL,int MaxIterationNum=10,int MinClusterSampleNumber=2);
	template <class T> static double norm(T* X,int Dim);
	template <class T1,class T2> static int FindClosestPoint(T1* pPointSet,int NumPoints,int nDim,T2* QueryPoint);
	template <class T1,class T2> static void GaussianFiltering(T1* pSrcArray,T2* pDstArray,int NumPoints,int nChannels,int size,double sigma);
};

template <class T>
void CStochastic::GetMeanVar(T* signal,int length,double* mean,double* var)
{
	double m_mean=0,m_var=0;

	int i;
	for (i=0;i<length;i++)
		m_mean+=signal[i];
	m_mean/=length;
	for (i=0;i<length;i++)
		m_var+=(signal[i]-m_mean)*(signal[i]-m_mean);
	m_var/=length-1;
	*mean=m_mean;
	*var=m_var;
}

template <class T>
T CStochastic::sum(int NumData, T* pData)
{
	T sum=0;
	int i;
	for(i=0;i<NumData;i++)
		sum+=pData[i];
	return sum;
}

template <class T>
void CStochastic::Normalize(int NumData,T* pData)
{
	int i;
	T Sum;
	Sum=sum(NumData,pData);
	for(i=0;i<NumData;i++)
		pData[i]/=Sum;
}

template <class T>
T CStochastic::mean(int NumData,T* pData)
{
	return sum(NumData,pData)/NumData;
}

////////////////////////////////////////////////////////////
// sort data in descending order
template <class T>
void CStochastic::sort(int Number,T* pData,int *pIndex,SortType m_SortType)
{
	int i,j,offset_extreme,*flag;
	double extreme;
	flag=new int[Number];
	memset(flag,0,sizeof(int)*Number);
	for(i=0;i<Number;i++)
	{
		if(m_SortType==SortDescending)
			extreme=-1E100;
		else
			extreme=1E100;
		offset_extreme=0;
		for(j=0;j<Number;j++)
		{
			if(flag[j]==1)
				continue;
			if( (m_SortType==SortDescending && extreme<pData[j]) || (m_SortType==SortAscending && extreme>pData[j]))
			{
				extreme=pData[j];
				offset_extreme=j;
			}
		}
		pIndex[i]=offset_extreme;
		flag[offset_extreme]=1;
	}
	delete flag;
}

template <class T>
T CStochastic::Min(int NumData,T* pData)
{
	int i;
	T result=pData[0];
	for(i=1;i<NumData;i++)
		result=__min(result,pData[i]);
	return result;
}

template <class T>
T CStochastic::Min(int NumData,T* pData1,T* pData2)
{
	int i;
	T result=pData1[0]+pData2[0];
	for(i=1;i<NumData;i++)
		result=__min(result,pData1[i]+pData2[i]);
	return result;
}