
#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include "project.h"

using namespace std;

template <class T>
class Vector
{
protected:
	int nDim;
	T* pData;
public:
	Vector(void);
	Vector(int ndim,const T *data=NULL);
	Vector(const Vector<T>& vect);
	~Vector(void);
	void releaseData();
	void allocate(int ndim);
	void allocate(const Vector<T>& vect){allocate(vect.nDim);};
	void copyData(const Vector<T>& vect);
	void dimcheck(const Vector<T>& vect) const;