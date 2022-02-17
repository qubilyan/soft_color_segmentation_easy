#pragma once
#include "stdio.h"
#include <vector>

template <class T>
void _Release1DBuffer(T* pBuffer)
{
	if(pBuffer!=NULL)
		delete []pBuffer;
	pBuffer=NULL;
}

temp