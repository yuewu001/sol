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
    static const double init_tbd = 1e10;
	//learning rate
	static const double init_eta = 0.01;
	//l1 regularization
	static const double init_lambda = 0.0;
	//sparse soft threshold when counting zero-weights
	static const double init_sparse_soft_thresh = 1e-5;
	//truncate gradients every K steps
	static const int init_k = 10;
	//gamma in enchanced RDA
	static const double init_gamma = 5000;
	//rou in enchanced RDA
	static const double init_rou = 0.005;
	//delta in adaptive algorithms
	static const double init_delta = 10;
	//whether to randomize the order of elements
	static const bool init_is_random = false;
	//number of rounds to run to test the performances of the algorithm
	static const int init_round_num = 1;

	static const enum_Loss_Type init_loss_type = Loss_Type_Logit;
	static const int init_data_type = DataSet_Type_BC | DataSet_MNIST;
	static const enum_Opti_Method init_opti_method = Opti_Ada_RDA;

	//trying the optimal parameters
	static const double init_eta_max = 1;
	static const double init_eta_min = 1e-10;
	static const double init_eta_step = 10;
	static const double init_delta_max = 10;
	static const double init_delta_min = 0.1;
	static const double init_delta_step = 10;

	////////////////////Data Set Reader Parameters///////////////////////////
	static const int init_chunk_size = 256;
	static const int init_buf_size = 2;

	static const std::string init_tmp_file = "tmp~";
}
