/*************************************************************************
	> File Name: config.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 5/5/2014 4:07:53 PM
	> Functions: configurations of the toolbox
 ************************************************************************/

#ifndef HEADER_CONFIG_ML
#define HEADER_CONFIG_ML

#include <stdint.h>

namespace BOC{

#define IndexType uint32_t
	//sparse soft threshold when counting zero-weights
	static const float init_sparse_soft_thresh = (float)(1e-5);
}

#endif


