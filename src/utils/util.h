/*************************************************************************
	> File Name: util.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 8/19/2013 Monday 2:17:56 PM
	> Functions:
	************************************************************************/
#ifndef HEADER_UTIL
#define HEADER_UTIL

#include <cstring>
#include <numeric>
#include <math.h>
#include <ctype.h>

#if WIN32
#include <direct.h>
#include <io.h>
#include <windows.h>
#define SOL_ACCESS(x) _access(x,0)
#else
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>
#define SOL_ACCESS(x) access(x,F_OK)
#endif

#include <cstdlib>
#include <iostream>
#include <sstream>

#define DELETE_ARRAY(pointer) \
if (pointer != NULL){ \
	delete[](pointer); \
	pointer = NULL; \
}

#define DELETE_POINTER(pointer) \
if (pointer != NULL){ \
	delete (pointer); \
	pointer = NULL; \
}

//check if the argument is valid and throw exception otherwise
#define INVALID_ARGUMENT_EXCEPTION(argu, condition, correctCondition) \
    if ((condition) == false){ \
        ostringstream oss; \
        oss<<"Argument "<<#argu<<" must be "<<correctCondition<<". Input is "<< argu; \
        throw invalid_argument(oss.str().c_str()); \
    }


template <typename T>
inline char Sgn(T x) {
	if (x > 0) return 1;
	else if (x < 0) return -1;
	else  return 0;
}

//#define ABS(x) (x > 0 ? x : -x)
template <typename T>
inline T ABS(T x) {
	return x > 0 ? x : -x;
}

template <typename T>
inline float Average(const T* data, int dim) {
	return std::accumulate(data, data + dim, 0.f) / (float)dim;
}

template <typename T>
float Variance(const T* data, int dim) {
	if (dim <= 1)
		return 0;
	float ave = std::accumulate(data, data + dim, 0.f) / (float)dim;
	double var(0);
	for (int i = 0; i < dim; i++)
		var += (data[i] - ave) * (data[i] - ave);
	return (float)(sqrt(var / (dim - 1)));
}

inline float trunc_weight(float w, float gravity){
	if (w > 0)
		return (gravity < w) ? w - gravity : 0.f;
	else
		return (gravity < -w) ? w + gravity : 0.f;
}
inline float trunc_weight2(float w, float gravity){
	if (w > 0)
		return (gravity < w) ? -gravity : -w;
	else
		return (gravity < -w) ? gravity : -w;
}

inline void ToUpperCase(char* c_str) {
	size_t i = 0;
	while (1){
		if (c_str[i] >= 'a' && c_str[i] <= 'z'){
			c_str[i] -= 32;
			i++;
		}
		else if (c_str[i] >= 'A' && c_str[i] <= 'Z'){
			i++;
		}
		else
			break;
	}
}
inline void ToLowerCase(char* c_str) {
	size_t i = 0;
	while (1){
		if (c_str[i] >= 'a' && c_str[i] <= 'z'){
			i++;
		}
		else if (c_str[i] >= 'A' && c_str[i] <= 'Z'){
			c_str[i] += 32;
			i++;
		}
		else
			break;
	}
}

inline void ToUpperCase(string &str) {
	string dst_str;
	int len = (int)(str.length());
	for (int i = 0; i < len; i++)
		dst_str.push_back(toupper(str[i]));
	std::swap(str, dst_str);
}

inline void ToLowerCase(string &str) {
	string dst_str;
	int len = (int)(str.length());
	for (int i = 0; i < len; i++)
		dst_str.push_back(tolower(str[i]));
	std::swap(str, dst_str);
}

inline bool rename_file(const string& src_filename, const string& dst_filename){
	string cmd;
	if (SOL_ACCESS(dst_filename.c_str()) == 0){
		//del the original cache_file
#if WIN32
		cmd = "del \"" + dst_filename + "\"";
#else
		cmd = "rm \"" + dst_filename + "\"";
#endif
		if (system(cmd.c_str()) != 0){
			std::cerr << "del original cache file failed!" << std::endl;
			return false;
		}
	}
	//rename
#if WIN32
	cmd = "ren \"";
	cmd = cmd + src_filename + "\" \"";
	//in windows, the second parameter of ren should not include path
	cmd = cmd + dst_filename.substr(dst_filename.find_last_of("/\\") + 1) + "\"";
#else
	cmd = "mv \"";
	cmd = cmd + src_filename + "\" \"";
	cmd = cmd + dst_filename + "\"";
#endif

	if (system(cmd.c_str()) != 0){
		std::cerr << "rename cahe file name failed!" << std::endl;
		return false;
	}
	return true;
}

inline double get_current_time(){
#if _WIN32
	return GetTickCount() / 1000.0;
#else
	struct timeval tim;
	gettimeofday(&tim, NULL);
	return tim.tv_sec + tim.tv_usec / 1000000.0;
#endif
}

template <typename T1, typename T2>
void QuickSort(T1 *a, T2 *b, size_t low, size_t high){ // from small to great
	size_t i = low;
	size_t j = high;
	T1 temp = a[low]; // select the first element as the indicator
	T2 temp_ind = b[low];

	while (i < j) {
		while ((i < j) && (temp < a[j])){ // scan right side
			j--;
		}
		if (i < j) {
			a[i] = a[j];
			b[i] = b[j];
			i++;
		}

		while (i < j && (a[i] < temp)){       // scan left side
			i++;
		}
		if (i < j) {
			a[j] = a[i];
			b[j] = b[i];
			j--;
		}
	}
	a[i] = temp;
	b[i] = temp_ind;

	if (low < i) {
		QuickSort(a, b, low, i - 1);  // sort left subset
	}
	if (i < high) {
		QuickSort(a, b, j + 1, high);  // sort right subset
	}
}
#endif
