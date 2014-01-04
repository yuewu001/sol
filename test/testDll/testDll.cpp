/*************************************************************************
  > File Name: testDll.cpp
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: 2014/1/3 9:25:01
  > Functions: Test if dynamic dll works
 ************************************************************************/

#include "SOL_interface.h"

#include <iostream>
#include <cstring>
#include <vector>
#include <cstdio>

using namespace std;
void release(long &dataset){
	sol_release_dataset(dataset);
	dataset = 0;
}
void release(long &dataset, long &loss_func){
	sol_release_dataset(dataset);
	sol_release_loss(loss_func);
	dataset = loss_func = 0;
}
void release(long &dataset, long &loss_func, long &opti){
	sol_release_dataset(dataset);
	sol_release_loss(loss_func);
	sol_release_optimizer(opti);
	dataset = loss_func = opti = 0;
}

int main(int argc, const char** args){
    printf("example: testDll [train_file test_file]\n");
	const char* train_file = "../../data/a7a/a7a";
	const char* test_file = "../../data/a7a/a7a.t";
    if(argc == 3){
        train_file = args[1];
        test_file = args[2];
    }

	vector<const char*> args_vec;
	args_vec.push_back("-opt");
	args_vec.push_back("AROW");

	long dataset = sol_init_dataset(train_file,"","libsvm",-1,-1);
	if (dataset == 0 ) {
		release(dataset);
		return -1;
	}
	long loss_func = sol_init_loss("hinge");
	if (loss_func == 0){
		release(dataset,loss_func);
		return -1;
	}
	long opti = sol_init_optimizer(dataset, loss_func,args_vec.size(), &(args_vec[0]));
	if  (opti == 0){
		release(dataset, loss_func, opti);
		return -1;
	}

	float l_err(0), var_err(0), sparse_rate(0), time_cost(0);
	sol_train(opti,&l_err, &var_err, &sparse_rate, &time_cost);
	

	long t_dataset = sol_init_dataset(test_file,"","libsvm",-1,-1);
	if (t_dataset != 0){
		float t_errRate(0), t_cost(0);
		sol_test(opti,t_dataset,&t_errRate, &t_cost);
		sol_release_dataset(t_dataset);
		printf("Learn error rate: %.2f +/- %.2f %%\n",l_err * 100, var_err * 100);
		printf("Test error rate: %.2f %%\n",t_errRate* 100); 
		printf("Sparsification Rate: %.2f %%\n", sparse_rate * 100);
		printf("Learning time: %.3f s\n", time_cost);
		printf("Test time: %.3f s\n", t_cost);
	}
	else{
		printf("Learn error rate: %.2f +/- %.2f %%\n",l_err * 100, var_err * 100);
		printf("Sparsification Rate: %.2f %%\n", sparse_rate * 100);
		printf("Learning time: %.3f s\n", time_cost);
	}
	release(dataset, loss_func, opti);
	return 0;
}
