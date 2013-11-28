/*************************************************************************
> File Name: Params.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: Thu 26 Sep 2013 05:51:05 PM SGT
> Functions: Class for Parsing parameters
************************************************************************/

#pragma once

#include "common/global.h"
#include "data/parser.h"

#include <string>
#include <map>

using std::string;
using std::map;

namespace SOL
{
	class Params
	{
	public:
		//input data
		string fileName; //source file name
		string cache_fileName; //cached file name
		string test_fileName; //test file name
		string test_cache_fileName; //cached test file name

		//dataset type
		int data_type;
		//optimization method
		enum_Opti_Method opti_method;
		//loss function type
		enum_Loss_Type loss_type;

		int passNum;

		//optimzation parameters
		float eta; //learning rate
		float lambda; //for l1 regularization
		int K; //for STG method
		float gamma_rou; //for RDA
		float delta; //for Ada-
		float r; //for AROW
		float phi; //for SCW

		int buf_size; //number of chunks in dataset 

		size_t initial_t;
		float power_t; 
		bool is_learn_best_param; //whether learn best parameter

	public:
		Params(){}
		Params(int argc, char** args);

		void Parse(int argc, char** args);
		void Help();

	private:
		//default parameter settings
		void Default();
		void ParseOptiMethod(char *str_method);
		void ParseDataType(char *str_data_type);
		enum_Loss_Type GetLossType(char *str_type);
	};
}
