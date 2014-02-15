/*************************************************************************
	> File Name: main.cpp
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2013/9/20 13:18:02
	> Functions: 
 ************************************************************************/
#include "Params.h"

#include "utils/util.h"

#include "io/DataSet.h"
#include "io/MPDataSet.h"
#include "io/sol_io.h"
#include "loss/sol_loss.h"
#include "algorithms/sol_algorithms.h"

#include <string>
#include <iostream>
#include <fstream>
#include <cmath>


using namespace std;
using namespace SOL;

#define FeatType float
#define LabelType char
//to be defined in sol_interface.cpp
template <typename T1, typename T2>
Optimizer<T1,T2>* GetOptimizer(const Params &param, DataSet<T1,T2> &dataset, LossFunction<T1,T2> &lossFun);

int main(int argc, const char** args) {
	//check memory leak in VC++
#if defined(_MSC_VER) && defined(_DEBUG)
	int tmpFlag = _CrtSetDbgFlag( _CRTDBG_REPORT_FLAG );
	tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
	_CrtSetDbgFlag( tmpFlag );
	//_CrtSetBreakAlloc(2208);
#endif
	Params param;
	if (param.Parse(argc, args) == false){
		return -1;
	}
	if (param.cache_fileName.size() == 0 && param.fileName.length() == 0){
		cerr<<"ERROR: you must specify the training data\n"<<endl;
		param.Help();
		return -1;
	}

	LossFunction<FeatType, LabelType> *lossFunc = GetLossFunc<FeatType, LabelType>(param.str_loss);
	if(lossFunc == NULL)
		return -1;

	DataSet<FeatType,LabelType>* pDataset = getDataSet<FeatType, LabelType>(param.passNum, param.buf_size, param.str_mp_type,param.mp_buf_size);

	if (pDataset->Load(param.fileName, param.cache_fileName) == false){
		cerr<<"ERROR: Load dataset "<<param.fileName<<" failed!"<<endl;
		delete pDataset;
		delete lossFunc;
		return -1;
	}

	Optimizer<FeatType, LabelType> *opti = GetOptimizer(param,*pDataset,*lossFunc);
	if (opti == NULL) {
		delete lossFunc;
		return -1;
	}

	opti->SetParameter(param.lambda,param.eta, param.power_t, param.initial_t);
	if (param.is_normalize == true)
		opti->SetNormalize(param.is_normalize);

	//learn the best parameters
	if (param.is_learn_best_param == true){
		opti->BestParameter();
	}
	else{
		opti->PrintOptInfo();

		float l_errRate(0), l_varErr(0);	//learning error rate
		float sparseRate(0);

		//learning the model
		double time1 = get_current_time();

		opti->Learn(l_errRate,l_varErr,sparseRate);
		
		if (param.out_readable_model.length() > 0){
			opti->SaveModel(param.out_readable_model);
		}

		double time2 = get_current_time();

		printf("data number: %lu\n",pDataset->size());
		printf("Learn error rate: %.2f +/- %.2f %%\n",l_errRate * 100, l_varErr * 100);

		double time3 = 0;
		//test the model
		bool is_test = param.test_cache_fileName.length() > 0 || param.test_fileName.length() > 0;
		if ( is_test) {
			DataSet<FeatType, LabelType> testset(1,param.buf_size);
			if (testset.Load(param.test_fileName, param.test_cache_fileName) == true) {
				float t_errRate(0);	//test error rate
				t_errRate = opti->Test(testset);
				time3 = get_current_time();

				printf("Test error rate: %.2f %%\n",t_errRate * 100); 
			}
			else
				cout<<"load test set failed!"<<endl;
		}

		printf("Non-Zero weight number: %u\n", opti->GetNonZeroNum());
		printf("Sparsification Rate: %g %%\n", sparseRate * 100);
		printf("Learning time: %.3f s\n", (float)(time2 - time1));
		if (is_test)
			printf("Test time: %.3f s\n", (float)(time3 - time2));
	}

	delete lossFunc;
	delete opti;

	return 0;
}
