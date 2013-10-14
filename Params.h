/*************************************************************************
> File Name: Params.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: Thu 26 Sep 2013 05:51:05 PM SGT
> Functions: Class for Parsing parameters
************************************************************************/

#pragma once

#include "common/global.h"

#include <string>
using namespace std;

namespace SOL
{
	class Params
	{
	public:
		//input data
		string fileName; //source file name
		string labelFile; //label file name
		string cache_fileName; //cached file name
		string test_fileName; //test file name
		string test_label_fileName; //test label file name
		string test_cache_fileName; //cached test file name

		//dataset type
		int data_type;
		//optimization method
		enum_Opti_Method opti_method;
		//loss function type
		enum_Loss_Type loss_type;

		int passNum;

		//optimzation parameters
		double eta; //learning rate
		double lambda; //for l1 regularization
		int K; //for STG method
		double theta; //for trucated gradient
		double gamma; //fro RDA
		double rou; //for RDA
		double delta; //for Ada-
        double r; //for AROW

		int digit_1;	//digit number to be classified in MINST dataset
		int digit_2;	//digit number to be classified in MINST dataset

		int buf_size; //number of chunks in dataset 

		bool is_rand;	//whether to randomize the order of data
		int round_num;	//number of rounds to run to test the performance

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
