/*************************************************************************
	> File Name: test.cpp
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2013/9/20 13:18:02
	> Functions: 
 ************************************************************************/

#include "data/DataSet.h"
#include "data/libsvmread.h"

#include "optimizer/SGD.h"
#include "optimizer/STG.h"
#include "optimizer/RDA_L1.h"
#include "optimizer/FOBOS.h"
#include "optimizer/ASM_L1.h"

#include "loss/LogisticLoss.h"
#include "loss/HingeLoss.h"
#include "loss/SquareLoss.h"

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
    //Params param(argc, args);

    if (PrepareDataset(param) == false)
        return -1;
    DataSet<FeatType, LabelType> dataset(param.passNum,1);
    if (dataset.Load(param.cache_fileName) == false)
		return -1;
    LossFunction<FeatType, LabelType> *lossFunc = GetLossFunc<FeatType, LabelType>(param);
	if(lossFunc == NULL)
		return -1;
	Optimizer<FeatType, LabelType> *opti = GetOptimizer(param,dataset,*lossFunc);
	if (opti == NULL)
		return -1;

	opti->SetParameter(param.lambda,param.eta);
	opti->RandomOrder(param.is_rand);

	//learn the best parameters
    opti->BestParameter();

    double l_errRate(0), l_varErr(0);	//learning error rate
	double sparseRate(0);

	//learning the model
    clock_t time1 = clock();

	opti->Learn(l_errRate,l_varErr,sparseRate,param.round_num);

    clock_t time2 = clock();

	printf("--------------------------------------------------\n");
	printf("Algorithm: %s\n",opti->Id_Str().c_str());
    printf("learning error rate %.2f +/- %.2f %%\n",l_errRate * 100, l_varErr * 100);

	clock_t time3 = 0;
	//test the model
    bool is_test = param.test_cache_fileName.length() > 0;
	if ( is_test && SOL_ACCESS(param.test_cache_fileName.c_str()) == 0)
	{
		DataSet<FeatType, LabelType> testset(1,param.buf_size);
		if (testset.Load(param.test_cache_fileName) == true)
		{
			double t_errRate(0);	//test error rate
			t_errRate = opti->Test(testset);
			time3 = clock();

			printf("Test error rate %.2f %%\n",t_errRate * 100); 
		}
        else
            cout<<"load test set failed!"<<endl;
	}

	printf("Sparsification Rate: %.2f %%\n", sparseRate * 100);
    printf("Learning time: %.3f s\n", (float)(time2 - time1) / CLOCKS_PER_SEC);
    if (is_test)
        printf("Test time: %.3f s\n", (float)(time3 - time2) / CLOCKS_PER_SEC);

    delete lossFunc;
    delete opti;
    delete []argv;

    return 0;
}
bool PrepareDataset(const Params &param)
{
    //check if the cache file exists
	if (param.cache_fileName.length() != 0 &&
    SOL_ACCESS(param.cache_fileName.c_str()) != 0) //cache_file not exist
    {
        cout<<"Converting training dataset to desired form..."<<flush;
        clock_t start = clock();
        bool status = DataSet_Converter<FeatType, LabelType>::Convert(param);
        clock_t end = clock();
        cout<<"finished\n";
        cout<<"time elapsed: "<<(float)(end - start) / CLOCKS_PER_SEC<<" s"<<endl;
		if (status == false)
			return false;
	}
	//check if the cache file exists
	if (param.test_cache_fileName.length() != 0 &&
    SOL_ACCESS(param.test_cache_fileName.c_str()) != 0) //cache_file not exist
    {
        cout<<"Converting test dataset to desired form..."<<flush;
        clock_t start = clock();
        bool status = DataSet_Converter<FeatType, LabelType>::Convert(param,true);
        clock_t end = clock();
        cout<<"finished\n";
        cout<<"time elapsed: "<<(float)(end - start) / CLOCKS_PER_SEC<<" s"<<endl;
		if (status == false)
			return false;
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
			return NULL;
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
                RDA_L1<T1,T2> *opti = new RDA_L1<T1,T2>(dataset,lossFunc,false);
                opti->SetParameterEx(param.lambda,param.rou);
                return opti;
                break;
            }
        case Opti_RDA_E:
            {
                RDA_L1<T1,T2> *opti = new RDA_L1<T1,T2>(dataset,lossFunc,true);
                opti->SetParameterEx(param.lambda,param.rou);
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

void FakeInput(int &argc, char **args, char** &argv)
{
    /*
    char* fileName = "../data/rcv1.train";
    //char* cache_fileName = "../data/cache_rcv1";
    char* testFileName = "../data/rcv1.test";
    */
#if WIN32
    char* fileName = "D:/Skydrive/Coding/Projects/SOL/data/MNIST/train-images-idx3-ubyte";
    char* labelFile = "D:/Skydrive/Coding/Projects/SOL/data/MNIST/train-labels-idx1-ubyte";
    char* testFileName = "D:/Skydrive/Coding/Projects/SOL/data/MNIST/t10k-images-idx3-ubyte";
    char* testLabelFile = "D:/Skydrive/Coding/Projects/SOL/data/MNIST/t10k-labels-idx1-ubyte";
#else
	char* fileName = "/home/matthew/SOL/data/MNIST/train-images-idx3-ubyte";
    char* labelFile = "/home/matthew/SOL/data/MNIST/train-labels-idx1-ubyte";
    char* testFileName = "/home/matthew/SOL/data/MNIST/t10k-images-idx3-ubyte";
    char* testLabelFile = "/home/matthew/SOL/data/MNIST/t10k-labels-idx1-ubyte";
#endif
    int app_len = 8;
    argv = new char*[argc + app_len];
    for (int i = 0; i < argc; i++)
        argv[i] = args[i];

    argv[argc] = "-i";
    argv[argc + 1] = fileName;
    argv[argc + 2] = "-t";
    argv[argc + 3] = testFileName;
//    argv[argc + 4] = "-opt";
 //   argv[argc + 5] = "RDA_E";

    argv[argc + 4] = "-il";
    argv[argc + 5] = labelFile;
    argv[argc + 6] = "-tl";
    argv[argc + 7] = testLabelFile;

    argc += app_len;
}


