/*************************************************************************
	> File Name: BOC.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 5/14/2014 10:36:59 PM
	> Functions: class for the toolbox
 ************************************************************************/
#ifndef HEADER_BOC_TOOLBOX
#define HEADER_BOC_TOOLBOX

#include "utils/Params.h"
#include "utils/util.h"
#include "io/io_header.h"
#include "algorithms/algo_header.h"
#include "loss/loss_header.h"
#include "optimizers/opt_header.h"

#include <string>
#include <stdio.h>

namespace BOC{
template <typename FeatType, typename LabelType>
class LibBOC {
    protected:
        //information of algorithms
        std::string algoInfo;
        //information of loss functions
        std::string lossInfo;
        //information of dataset
		std::string datasetInfo;
        //informationi of io
		std::string ioInfo;
        //information of optimizers
		std::string optInfo;

        //loss function
        LossFunction<FeatType, LabelType> *pLossFunc;
        //dataset
        DataSet<FeatType, LabelType>* pDataset;
        //training model
        OnlineModel<FeatType, LabelType> * pOnlineModel;
        LearnModel<FeatType, LabelType> * pModel;
        //optimizer
        Optimizer<FeatType, LabelType> * pOpti;

        Params *pParam;

    public:
        LibBOC(){
            this->pLossFunc = NULL;
            this->pDataset  = NULL;
            this->pModel    = NULL;
            this->pOnlineModel    = NULL;
            this->pOpti = NULL;
            this->pParam = NULL;
            ModelInfo<FeatType, LabelType>::GetModelInfo(this->algoInfo);
			LossInfo<FeatType, LabelType>::GetLossInfo(this->lossInfo);
			DataSetInfo<FeatType, LabelType>::GetDataSetInfo(this->datasetInfo);
			IOInfo<FeatType, LabelType>::GetIOInfo(this->ioInfo);
			OptInfo<FeatType, LabelType>::GetOptInfo(this->optInfo);
        }

        ~LibBOC(){
            this->Release();
        }

    protected:
        void Release(){
            DELETE_POINTER(this->pLossFunc);
            DELETE_POINTER(this->pDataset);
            DELETE_POINTER(this->pModel);
            DELETE_POINTER(this->pOpti);
			this->pOnlineModel = NULL;
            this->pParam = NULL;
        }

        /**
         * @Synopsis Show information of the toolbox
         */
    public:
        inline void ShowAlgoInfo() const {
            printf("Algorithms:\n%s",algoInfo.c_str());
        }
        inline void ShowLossInfo() const {
            printf("Loss Functions:\n%s",lossInfo.c_str());
        }

    protected:
        inline int InitLoss(Params &param){
			this->pLossFunc = (LossFunction<FeatType, LabelType>*)
				Registry::CreateObject(param.StringValue("-loss"));
			if (this->pLossFunc == NULL)
				return -1;
			return 0;
		}

		inline int InitDataSet(Params &param){
			this->pDataset = (DataSet<FeatType, LabelType>*)Registry::CreateObject(
				param.StringValue("-dt"), (void*)param.IntValue("-passes"),
				(void*)param.IntValue("-bs"), (void*)param.IntValue("-cs"));
			if (this->pDataset == NULL){
				return -2;
			}

			if (this->pDataset->Load(param.StringValue("-i"), param.StringValue("-c"),
				param.StringValue("-df")) == false){
				cerr << "ERROR: Load dataset " << param.StringValue("-i") << " failed!" << endl;
				return -3;
			}
			return 0;
		}

		inline int InitModel(Params &param){
			//check model type
			string algo = param.StringValue("-algo");
			ToUpperCase(algo);
			this->pModel = (LearnModel<FeatType, LabelType>*)Registry::CreateObject(algo, this->pLossFunc);
			const string& modelType = this->pModel->GetModelType();
			if (modelType == "online"){
				this->pOnlineModel = (OnlineModel<FeatType, LabelType>*)this->pModel;
				if (this->pOnlineModel == NULL){
					return -4;
				}
			}
			else{
				fprintf(stderr, "Error: unsupported model type %s!", modelType.c_str());
				return -4;
			}

			this->pModel->SetParameter(param);

			return 0;
		}

		inline int InitOptimizer(Params &param){
			string optType = param.StringValue("-opt");
			ToLowerCase(optType);
			this->pOpti = (Optimizer<FeatType, LabelType>*)
				Registry::CreateObject(optType, this->pOnlineModel, this->pDataset);

			if (this->pOpti == NULL) {
				return -5;
			}
			return 0;
		}

public:
	int Initialize(Params &param){
		this->Release();
		int errCode = 0;
		errCode = this->InitDataSet(param);
		errCode = this->InitLoss(param);
		errCode = this->InitModel(param);
		errCode = this->InitOptimizer(param);
		if (errCode != 0){
			return errCode;
		}
		this->pParam = &param;
		return errCode;
	}

	int Run(){
		//learning the model
		double time1 = get_current_time();

		float l_errRate = this->pOpti->Train();

		if (this->pParam->StringValue("-or").length() > 0){
			this->pModel->SaveModel(this->pParam->StringValue("-or"));
		}

		double time2 = get_current_time();
		printf("data number: %lu\n", this->pDataset->size());
		printf("Learn error rate: %.2f %%\n", l_errRate * 100);
		printf("Learning time: %.3f s\n", (float)(time2 - time1));
		this->pModel->PrintModelInfo();

		//test
		double time3 = 0;
		//test the model
		bool is_test = this->pParam->StringValue("-tc").length() > 0 ||
			this->pParam->StringValue("-t").length() > 0;
		if (is_test) {
			OnlineDataSet<FeatType, LabelType> testset(1, this->pParam->IntValue("-bs"), this->pParam->IntValue("-cs"));
			if (testset.Load(this->pParam->StringValue("-t"), this->pParam->StringValue("-tc"),
				this->pParam->StringValue("-df")) == true) {
				float t_errRate(0);	//test error rate
				t_errRate = this->pOpti->Test(testset);
				time3 = get_current_time();

				printf("Test error rate: %.2f %%\n", t_errRate * 100);
			}
			else{
				fprintf(stderr, "load test set failed!");
			}
			printf("Test time: %.3f s\n", (float)(time3 - time2));
		}

		return 0;
	}
};

}

#endif

