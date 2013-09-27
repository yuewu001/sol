/*************************************************************************
	> File Name: test.cpp
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2013/9/20 13:18:02
	> Functions: 
 ************************************************************************/

#include "DataSet.h"
#include "libsvmread.h"

#include "SGD.h"
#include "STG.h"
#include "RDA_L1.h"
#include "FOBOS.h"
#include "ASM_L1.h"

#include "LogisticLoss.h"
#include "HingeLoss.h"
#include "SquareLoss.h"

#include "Params.h"
#include "DataSet_Converter.h"

#include <string>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>

using namespace std;
using namespace SOL;

#define FeatType float
#define LabelType char

bool PrepareDataset(const Params &param);
void FakeInput(int &argc, char **args, char** &argv);
template <typename T1, typename T2>
LossFunction<T1,T2>* GetLossFunc(const Params &param);
template <typename T1, typename T2>
Optimizer<T1,T2>* GetOptimizer(const Params &param, DataSet<T1,T2> &dataset, LossFunction<T1,T2> &lossFun);

int main(int argc, char** args)
{
    char** argv;
    FakeInput(argc, args, argv);
    Params param(argc, argv);

    DataSet<FeatType, LabelType> dataset(param.passNum,param.chunk_size);
    if (PrepareDataset(param) == false)
        return -1;

    LossFunction<FeatType, LabelType> *lossFunc = GetLossFunc<FeatType, LabelType>(param);

    clock_t start = clock();
    dataset.Load(param.cache_fileName);
    Optimizer<FeatType, LabelType> *opti = GetOptimizer(param,dataset,*lossFunc);
//    opti->SetParameter(-1,0.1);
    opti->BestParameter();

    double errRate, varErr, sparseRate;
    opti->Learn(errRate,varErr,sparseRate,1);
    printf("Error Rate %f%%:+/-%f%%\n",errRate * 100, varErr * 100);
    printf("Sparse Rate %f%%\n",sparseRate * 100);

    clock_t end = clock();
    cout<<"time elapsed: "<<(float)(end - start) / CLOCKS_PER_SEC<<" s"<<endl;

    delete lossFunc;
    delete opti;
    delete []argv;

    return 0;
}
void FakeInput(int &argc, char **args, char** &argv)
{
    /*
    char* fileName = "/home/matthew/SOL/data/rcv1.train";
    char* cache_fileName = "/home/matthew/SOL/data/cache_rcv1";
    */

    char* fileName = "/home/matthew/SOL/data/MNIST/train-images-idx3-ubyte";
    char* labelFile = "/home/matthew/SOL/data/MNIST/train-labels-idx1-ubyte";
    char* testFileName = "/home/matthew/SOL/data/MNIST/t10k-images-idx3-ubyte";
    char* testLabelFile = "/home/matthew/SOL/data/MNIST/t10k-labels-idx1-ubyte";

    int app_len = 10;
    argv = new char*[argc + app_len];
    for (int i = 0; i < argc; i++)
        argv[i] = args[i];

    argv[argc] = "-i";
    argv[argc + 1] = fileName;
    argv[argc + 2] = "-il";
    argv[argc + 3] = labelFile;
    argv[argc + 4] = "-t";
    argv[argc + 5] = testFileName;
    argv[argc + 6] = "tc";
    argv[argc + 7] = testLabelFile;
    argv[argc + 8] = "-dt";
    argv[argc + 9] = "MNIST";
    argc += app_len;
}

bool PrepareDataset(const Params &param)
{
    //check if the cache file exists
    if (SOL_ACCESS(param.cache_fileName.c_str()) != 0) //cache_file not exist
    {
        cout<<"Converting the dataset to desired form..."<<flush;
        clock_t start = clock();
        bool status = DataSet_Converter::Convert(param);
        clock_t end = clock();
        cout<<"finished\n";
        cout<<"time elapsed: "<<(float)(end - start) / CLOCKS_PER_SEC<<" s"<<endl;
        return status;
    }
    return true;
}

template <typename T1, typename T2>
LossFunction<T1,T2>* GetLossFunc(const Params &param)
{
    switch(param.loss_type)
    {
        case Loss_Type_Hinge:
            return new HingeLoss<T1,T2>();
        case Loss_Type_Logit:
            return new LogisticLoss<T1,T2>();
        case Loss_Type_Square:
            return new SquareLoss<T1,T2>();
        default:
            cout<<"Unrecognized Loss function!"<<endl;
            exit(0);
    }
}

template <typename T1, typename T2>
Optimizer<T1,T2>* GetOptimizer(const Params &param, DataSet<T1,T2> &dataset, LossFunction<T1,T2> &lossFunc)
{
    switch(param.opti_method)
    {
        case Opti_SGD:
            {
            SGD<T1,T2> *opti = new SGD<T1,T2>(dataset,lossFunc);
            opti->SetParameter(param.lambda,param.eta);
            return opti;
            break;
            }
        case Opti_STG:
            {
                STG<T1,T2> *opti = new STG<T1,T2>(dataset,lossFunc);
                opti->SetParameterEx(param.lambda,param.K, param.eta,param.theta);
                return opti;
                break;
            }
        case Opti_RDA:
            {
                RDA_L1<T1,T2> *opti = new RDA_L1<T1,T2>(dataset,lossFunc);
                opti->SetParameterEx(param.lambda,param.gamma, param.rou);
                return opti;
                break;
            }
        case Opti_FOBOS:
            {
               FOBOS<T1,T2> *opti = new FOBOS<T1,T2>(dataset,lossFunc);
                opti->SetParameter(param.lambda,param.eta);
                return opti;
                break;
            }
        case Opti_Ada_RDA: 
            {
               ASM_L1<T1,T2> *opti = new ASM_L1<T1,T2>(dataset,lossFunc,ASM_Update_PSU);
                opti->SetParameterEx(param.lambda,param.delta,param.eta);
               return opti;
               break;
            }
        case Opti_Ada_FOBOS: 
            {
                ASM_L1<T1,T2> *opti = new ASM_L1<T1,T2>(dataset,lossFunc, ASM_Update_CMDU);
                opti->SetParameterEx(param.lambda,param.delta,param.eta);
                return opti;
                break;
            }
        default:
            break;
    }
    return NULL;
}
