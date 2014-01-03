/*************************************************************************
	> File Name: SOL_interface.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2014/1/2 22:10:53
	> Functions: interface design for dynamic and static library
 ************************************************************************/
#ifndef HEADER_SOL_INTERFACE_DLL
#define HEADER_SOL_INTERFACE_DLL

#if _WIN32
#ifndef DLL_HEADER
#define DLL_HEADER _declspec(dllimport)
#endif
#else
#define DLL_HEADER
#endif

#include "SOL_interface.h"
