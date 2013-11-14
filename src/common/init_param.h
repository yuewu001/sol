/*************************************************************************
> File Name: init_param.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/9/28 15:12:27
> Functions: init parameters
************************************************************************/

#pragma once
#include "global.h"
#include <string>

namespace SOL
{
//
	/////////////////////Optimizer Initalization parameters//////////////////
	//
    //value to be determined
    static const float init_tbd = 1e10;
	//learning rate
	static const float init_eta = 10;
	//l1 regularization
	static const float init_lambda = 0.0;
	//sparse soft threshold when counting zero-weights
	static const float init_sparse_soft_thresh = 1e-6;
	//truncate gradients every K steps
	static const int init_k = 10;
	//gamma in enchanced RDA
	static const float init_gamma = 5000;
	//rou in enchanced RDA
	static const float init_rou = 0.005;
	//delta in adaptive algorithms
	static const float init_delta = 10;
    //trade-off parameter in AROW
    static const int init_r = 1;
    //skip value in SVM2SGD
    static const int init_skip = 16;
    //t0 to avoid large parameter updates in the few updates
    static const size_t init_t0 = 10e4;

	static const enum_Loss_Type init_loss_type = Loss_Type_Logit;
	static const int init_data_type = DataSet_Type_BC | DataSet_LibSVM;
	static const enum_Opti_Method init_opti_method = Opti_STG;

	//trying the optimal parameters
	static const float init_eta_max = 100;
	static const float init_eta_min = 1e-6;
	static const float init_eta_step = 10;
	static const float init_delta_max = 10;
	static const float init_delta_min = 0.1;
	static const float init_delta_step = 10;

	////////////////////Data Set Reader Parameters///////////////////////////
	static const size_t init_chunk_size = 256;
	static const size_t init_buf_size = 2;

    //////////////////////Zlib Parameters/////////////////////////////
    static const int zlib_deflate_level = -1; // use default deflate level
    static const size_t zlib_buf_size = 16348; //default buffer size of zlib
}
