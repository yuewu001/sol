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
//information of algorithms
	static std::string algoInfo;
	//information of loss functions
	static std::string lossInfo;
	//informationi of io
	static std::string ioInfo;
	//information of optimizers
	static std::string optInfo;


	template <typename FeatType, typename LabelType>
	class LibBOC {
	protected:
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
			this->pDataset = NULL;
			this->pModel = NULL;
			this->pOnlineModel = NULL;
			this->pOpti = NULL;
			this->pParam = NULL;
			ModelInfo<FeatType, LabelType>::GetModelInfo(algoInfo);
			LossInfo<FeatType, LabelType>::GetLossInfo(lossInfo);
			IOInfo<FeatType, LabelType>::GetIOInfo(ioInfo);
			OptInfo<FeatType, LabelType>::GetOptInfo(optInfo);

			/*
			printf("%s", algoInfo.c_str());
			printf("%s", lossInfo.c_str());
			printf("%s", ioInfo.c_str());
			printf("%s", optInfo.c_str());
            */
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
			printf("Algorithms:\n%s", algoInfo.c_str());
		}
		inline void ShowLossInfo() const {
			printf("Loss Functions:\n%s", lossInfo.c_str());
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
			string dt_type = param.StringValue("-dt");
			ToLowerCase(dt_type);
			if (dt_type  == "online"){
				this->pDataset = new OnlineDataSet<FeatType, LabelType>(param.IntValue("-passes"));

				int buf_size = param.IntValue("-bs");
				int chunk_size = param.IntValue("-cs");
				const string& mp_buf_type = param.StringValue("-mbt");
				int mp_buf_size = param.IntValue("-mbs");
				((OnlineDataSet<FeatType, LabelType>*)this->pDataset)->ConfiBuffer(buf_size, chunk_size, mp_buf_type, mp_buf_size);
			}
			else if (dt_type == "batch"){
				this->pDataset = NULL;
			}
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
			this->pModel->PrintOptInfo();
			//learning the model
			double time1 = get_current_time();

			float l_errRate = this->pOpti->Train();

			if (this->pParam->StringValue("-or").length() > 0){
				this->pModel->SaveModel(this->pParam->StringValue("-or"));
			}

			double time2 = get_current_time();
			printf("\nData number: %lu\n", this->pDataset->size());
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

