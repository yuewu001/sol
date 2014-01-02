/*************************************************************************
	> File Name: SOL_interface.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2014/1/2 22:10:53
	> Functions: interface design for dynamic and static library
 ************************************************************************/
#ifndef HEADER_SOL_INTERFACE
#define HEADER_SOL_INTERFACE

#if _WIN32
#ifndef DLL_HEADER
#define DLL_HEADER _declspec(dllimport)
#endif
#else
#define DLL_HEADER
#endif
extern "C"{
/**
 *  sol_initialize : initialize an sol instance
 *
 * @Param:  argc: number of arguments
 * @Param:  args: char* array, command line
 *
 * @Returns:  pointer of the initialized object
 */

DLL_HEADER long sol_init_dataset(int argc, const char** args);
DLL_HEADER long sol_init_loss(int argc, const char** args);
DLL_HEADER long sol_init_optimizer(long dataset, long loss_func, int argc, const char** args);

/**
 *  sol_train: train a model
 *
 * @Param: optimizer: initialized sol instance
 * @Param:  t_errRate: average test error rate
 * @Param:  var_t_errRate: variance of the test error rate
 * @Param:  time_cost: time cost of the training
 *
 * @Returns:    
 */
DLL_HEADER void sol_train(long optimizer, float* l_errRate, float* var_errRate, float* sparse_rate, float* time_cost);

DLL_HEADER void sol_test(long optimizer, long test_dataset,  float* t_errRate, float* time_cost);

/**
 *  sol_release : release initialized objects
 *
 * @Param:  handler
 */
DLL_HEADER void sol_release_dataset(long dataset);
DLL_HEADER void sol_release_loss(long loss_func);
DLL_HEADER void sol_release_optimizer(long optimizer);
}

#endif
