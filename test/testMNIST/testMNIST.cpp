/*************************************************************************
  > File Name: testDll.cpp
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: 2014/1/3 9:25:01
  > Functions: Test how to use self-defined reader
 ************************************************************************/

#include "SOL_interface.h"
#include "MNISTReader.h"

#include <iostream>
#include <cstring>
#include <vector>
#include <cstdio>

using namespace std;
using namespace SOL;

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
    printf("example: testMNIST\n");
	const char* train_file = "../../data/MNIST/train-images-idx3-ubyte";
	const char* train_label = "../../data/MNIST/train-labels-idx1-ubyte";
	const char* test_file = "../../data/MNIST/t10k-images-idx3-ubyte";
	const char* test_label = "../../data/MNIST/t10k-labels-idx1-ubyte";

	DataReader<float, char>* reader = new MNISTReader<float, char>(train_file, train_label, 3, 8);
	if (reader == 0 ) {
		return -1;
	}
    long dataset = sol_init_dataset2((long)reader,-1,-1);
	if (dataset == 0 ) {
        delete reader;
		return -1;
	}
	long loss_func = sol_init_loss("hinge");
	if (loss_func == 0){
        delete reader;
		release(dataset,loss_func);
		return -1;
	}
    vector<const char*> args_vec;
    args_vec.push_back("-opt");
    args_vec.push_back("AROW-DA");
    args_vec.push_back("-l1");
    args_vec.push_back("1e-3");
    args_vec.push_back("-norm");


    long opti = sol_init_optimizer(dataset, loss_func,(int)(args_vec.size()), &(args_vec[0]));
    if  (opti == 0){
        delete reader;
        release(dataset, loss_func, opti);
        return -1;
    }

    float l_err(0), var_err(0), sparse_rate(0), time_cost(0);
    sol_train(opti,&l_err, &var_err, &sparse_rate, &time_cost);

    DataReader<float, char>* t_reader= new MNISTReader<float, char>(test_file, test_label, 3,8);
	long t_dataset = sol_init_dataset2((long)t_reader,-1,-1);

    if (t_dataset != 0){
        float t_errRate(0), t_cost(0);
        sol_test(opti,(long)t_dataset,&t_errRate, &t_cost);
        delete t_reader;
		release(t_dataset);
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
    delete reader;
    release(dataset, loss_func, opti);

    return 0;
}
