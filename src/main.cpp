/*************************************************************************
	> File Name: main.cpp
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2013/9/20 13:18:02
	> Functions:
	************************************************************************/
#include "utils/Params.h"

#include "utils/util.h"

#include "io/sol_io.h"
#include "loss/sol_loss.h"

#include "algorithms/om/olm/solm/sol_algorithms.h"
#include "optimizers/OnlineOptimizer.h"

#include <string>
#include <iostream>
#include <fstream>

using namespace std;
using namespace BOC;

#define FeatType float
#define LabelType char

#define ALGO SGD

int main(int argc, const char** args) {
	//check memory leak in VC++
#if defined(_MSC_VER) && defined(_DEBUG)
	int tmpFlag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
	tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
	_CrtSetDbgFlag(tmpFlag);
	//_CrtSetBreakAlloc(170);
#endif

	InitAlgorithms<FeatType,LabelType>();

	Params param;
	if (param.Parse(argc, args) == false){
		return -1;
	}

	try{
		cout << param.StringValue("-i") << endl;
		cout << param.StringValue("-t") << endl;
		cout << param.IntValue("-passes") << endl;
	}
	catch (exception &ex){
		cerr << ex.what();
	}

	if (param.StringValue("-c").size() == 0 && param.StringValue("-i").length() == 0){
		cerr << "ERROR: you must specify the training data\n" << endl;
		param.Help();
		return -1;
	}

	LossFunction<FeatType, LabelType> *lossFunc = GetLossFunc<FeatType, LabelType>(param.StringValue("-loss"));
	if (lossFunc == NULL)
		return -1;

	DataSet<FeatType, LabelType>* pDataset = getDataSet<FeatType, LabelType>(param.IntValue("-passes"), param.IntValue("-bs"),
		param.StringValue("-mpt"), param.IntValue("-mbs"));
	if (pDataset == NULL){
		delete lossFunc;
		return -1;
	}

	if (pDataset->Load(param.StringValue("-i"), param.StringValue("-c"), param.StringValue("-dt")) == false){
		cerr << "ERROR: Load dataset " << param.StringValue("-i") << " failed!" << endl;
		delete pDataset;
		delete lossFunc;
		return -1;
	}

//	SGD<FeatType, LabelType> * model = (SGD<FeatType, LabelType>*)Registry<FeatType, LabelType>::CreateObject("SGD",lossFunc);
	SparseOnlineLinearModel<FeatType, LabelType> * model = (SparseOnlineLinearModel<FeatType, LabelType>*)Registry::CreateObject("Ada_FOBOS",lossFunc);
	//ALGO<FeatType, LabelType> *model = new ALGO<FeatType, LabelType>(lossFunc);

	OnlineOptimizer<FeatType, LabelType> *opti = new OnlineOptimizer<FeatType, LabelType>(*model, *pDataset);
	if (opti == NULL) {
		delete lossFunc;
		return -1;
	}

	model->SetParameter(param);

	model->PrintOptInfo();

	//learning the model
	double time1 = get_current_time();

	float l_errRate = opti->Train();
	float sparseRate = model->GetSparseRate();

	if (param.StringValue("-or").length() > 0){
		model->SaveModel(param.StringValue("-or"));
	}

	double time2 = get_current_time();

	printf("data number: %lu\n", pDataset->size());
	printf("Learn error rate: %.2f %%\n", l_errRate * 100);

	double time3 = 0;
	//test the model
	bool is_test = param.StringValue("-tc").length() > 0 || param.StringValue("-t").length() > 0;
	if (is_test) {
		OnlineDataSet<FeatType, LabelType> testset(1, param.IntValue("-bs"), param.IntValue("-cs"));
		if (testset.Load( param.StringValue("-t"), param.StringValue("-tc"), param.StringValue("-dt")) == true) {
			float t_errRate(0);	//test error rate
			t_errRate = opti->Test(testset);
			time3 = get_current_time();

			printf("Test error rate: %.2f %%\n", t_errRate * 100);
		}
		else
			cout << "load test set failed!" << endl;
	}

	printf("Non-Zero weight number: %u\n", model->GetNonZeroNum());
	printf("Sparsification Rate: %g %%\n", sparseRate * 100);
	printf("Learning time: %.3f s\n", (float)(time2 - time1));
	if (is_test)
		printf("Test time: %.3f s\n", (float)(time3 - time2));
	printf("Update Times: %lu\n", opti->GetUpdateTimes());

	delete lossFunc;
	delete opti;

	return 0;
}
