/*************************************************************************
	> File Name: util.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 8/19/2013 Monday 2:17:56 PM
	> Functions: 
 ************************************************************************/
#pragma once

#include <cstring>
#include <numeric>
#include <cmath>
#include <ctype.h>

#if WIN32
#include <direct.h>
#include <io.h>
#define SOL_ACCESS(x) _access(x,0)
#else
#include <sys/stat.h>
#include <sys/types.h>
#define SOL_ACCESS(x) access(x,F_OK)
#endif


template <typename T>
char Sgn(T x) 
{
	if (x > 0) return 1;
	else if (x < 0) return -1;
	else  return 0;
}

template <typename T>
int MSB2LSB(T x)
{
	int y = x;
	int byteNum = sizeof(x);
	char *buf = new char[byteNum];
	char *buf1 = new char[byteNum];
	memcpy(buf, &x, byteNum);
	for (int i = 0; i < byteNum; i++)
		buf1[i] = buf[byteNum - 1 - i];
	memcpy(&y,buf1, byteNum);

	delete []buf;
	delete []buf1;
	return y;
}

template <typename T>
inline double Average(const T* data, int dim)
{
	return std::accumulate(data,data + dim, 0.0) / (double)dim;
}

template <typename T>
double Variance(const T* data, int dim)
{
	if (dim <= 1)
		return 0;
	double ave = std::accumulate(data, data + dim,0.0) / (double)dim;
	double var(0);
	for (int i = 0; i < dim; i++)
		var += (data[i] - ave) * (data[i] - ave);
	return sqrt(var / (dim - 1));
}


