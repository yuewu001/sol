/*************************************************************************
  > File Name: SOL_interface.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: 2014/1/2 22:10:53
  > Functions: interface design for dynamic and static library
 ************************************************************************/
#if _WIN32
#define DLL_HEADER _declspec(dllexport)
#endif

#include "Params.h"

#include "SOL_interface.h"

#include "algorithms/ModelInfo.h"

#include "utils/util.h"

#include "io/OnlineDataSet.h"
#include "io/sol_io.h"
#include "loss/sol_loss.h"
#include "algorithms/om/olm/solm/sol_algorithms.h"

#include <string>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;
using namespace SOL;

#define FeatType float
#define LabelType char

extern "C"{

/**
 * @Synopsis boc_init initialize the toolbox
 */
DLL_HEADER void boc_init(){

}

/**
 *  sol_initialize : initialize an sol instance
 *
 * @Param:  argc: number of arguments
 * @Param:  args: char* array, command line
 *
 * @Returns:  pointer of the initialized object
 */
long sol_init_dataset(const char* filename, const char* cache_filename, const char* dt_type, int passNum, int buf_size){
    if (filename == NULL && cache_filename == NULL){
        cerr << "no input is specified!" << endl;
        return 0;
    }
    OnlineDataSet<FeatType, LabelType> *dataset = new OnlineDataSet<FeatType, LabelType>(passNum, buf_size);
    if (dataset->Load(filename, cache_filename, dt_type) == false){
        cerr << "ERROR: Load dataset " << filename << " failed!" << endl;
        delete dataset;
        return 0;
    }
    return (long)dataset;
}

long sol_init_dataset2(long dt_reader, int passNum, int buf_size){
    return sol_init_dataset3(dt_reader, "", passNum, buf_size);
}

long sol_init_dataset3(long dt_reader, const char* cache_filename, int passNum, int buf_size){
    if (dt_reader == 0){
        cerr << "no reader is specified!" << endl;
        return 0;
    }
    OnlineDataSet<FeatType, LabelType> *dataset = new OnlineDataSet<FeatType, LabelType>(passNum, buf_size);
    if (dataset->Load((DataReader<FeatType, LabelType>*)dt_reader, cache_filename) == false){
        cerr << "ERROR: Load dataset failed!" << endl;
        delete dataset;
        return 0;
    }
    return (long)dataset;
}

long sol_init_loss(const char* loss_type){
    return (long)GetLossFunc<FeatType, LabelType>(loss_type);
}

long sol_init_optimizer(long dataset, long loss_func, int argc, const char** args){
    Params param;
    if (param.Parse(argc, args) == false){
        return 0;
    }
    Optimizer<FeatType, LabelType> *opti = GetOptimizer(param,
            *((DataSet<FeatType, LabelType>*)dataset),
            *((LossFunction<FeatType, LabelType>*)loss_func));
    if (opti == NULL) {
        return 0;
    }

    opti->SetParameter(param.lambda, param.eta, param.power_t, param.initial_t);
    if (param.is_normalize == true)
        opti->SetNormalize(param.is_normalize);
    if (param.is_learn_best_param == true){
        opti->BestParameter();
    }
    return (long)opti;
}

/**
 *  sol_train: train a model
 *
 * @Param: optimizer: initialized sol instance
 * @Param:  t_errRate: average test error rate
 * @Param:  var_t_errRate: variance of the test error rate
 * @Param:  sparse_rate: sparse rate of the model
 * @Param:  time_cost: time cost of the training
 *
 * @Returns:
 */
void sol_train(long optimizer, float* learn_errRate, float* var_l_errRate, float* sparse_rate, float* time_cost){
    Optimizer<FeatType, LabelType> *opti = (Optimizer<FeatType, LabelType> *)optimizer;
    if (opti == NULL)
        return;

    opti->PrintOptInfo();

    float l_errRate(0), l_varErr(0);	//learning error rate
    float sparseRate(0);

    //learning the model
    double time1 = get_current_time();

    opti->Learn(l_errRate, l_varErr, sparseRate);

    double time2 = get_current_time();

    *learn_errRate = l_errRate;
    *var_l_errRate = l_varErr;
    *sparse_rate = sparseRate;
    *time_cost = (float)(time2 - time1);
}

void sol_test(long optimizer, long test_dataset, float* t_errRate, float* time_cost){
    Optimizer<FeatType, LabelType> *opti = (Optimizer<FeatType, LabelType> *)optimizer;
    if (opti == NULL)
        return;
    double time2 = get_current_time();
    //test the model
    *t_errRate = opti->Test(*((DataSet<FeatType, LabelType>*)test_dataset));
    double time3 = get_current_time();

    *time_cost = (float)(time3 - time2);
}

/**
 *  sol_release : release initialized objects
 *
 * @Param:  handler
 */
void sol_release_dataset(long dataset){
    DataSet<FeatType, LabelType>* dt = (DataSet<FeatType, LabelType>*)dataset;
    if (dt != NULL)
        delete dt;
}


void sol_release_loss(long loss_func){
    LossFunction<FeatType, LabelType> * loss = (LossFunction<FeatType, LabelType> *)loss_func;
    if (loss != NULL)
        delete loss;
}

void sol_release_optimizer(long optimizer){
    Optimizer<FeatType, LabelType> *opti = (Optimizer<FeatType, LabelType> *)optimizer;
    if (opti != NULL)
        delete opti;
}
}
