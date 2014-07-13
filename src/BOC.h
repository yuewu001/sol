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
#include "utils/error.h"
#include "io/io_header.h"
#include "algorithms/algo_header.h"
#include "loss/loss_header.h"
#include "optimizers/opt_header.h"

#include <string>
#include <stdio.h>

namespace BOC{
//information of algorithms
	static std::string modelInfo;
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
			ModelInfo<FeatType, LabelType>::GetModelInfo(modelInfo);
			LossInfo<FeatType, LabelType>::GetLossInfo(lossInfo);
			IOInfo<FeatType, LabelType>::GetIOInfo(ioInfo);
			OptInfo<FeatType, LabelType>::GetOptInfo(optInfo);
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
		void ShowHelpInfo(Params& param){
			string helpItem = param.StringValue("-help");
			ToLowerCase(helpItem);
			if (helpItem == "model"){
				this->ShowModelInfo();
			}
			else if (helpItem == "optimizer"){
				this->ShowOptimizers();
			}
			else if (helpItem == "loss"){
				this->ShowLossInfo();
			}
			else if (helpItem == "io"){
				this->ShowIoInfo();
			}
		}

		inline void ShowModelInfo() const {
			printf("Models:\n%s\n", modelInfo.c_str());
		}
		inline void ShowLossInfo() const {
			printf("Loss Functions:\n%s\n", lossInfo.c_str());
		}

		inline void ShowOptimizers() const{
			printf("Optimizers: \n%s\n", optInfo.c_str());
		}
		inline void ShowIoInfo() const {
			printf("IO: \n%s\n", ioInfo.c_str());
		}

	protected:
		inline int InitLoss(Params &param){
			this->pLossFunc = (LossFunction<FeatType, LabelType>*)
				Registry::CreateObject(param.StringValue("-loss"));
			if (this->pLossFunc == NULL) {
				fprintf(stderr, "Error %d: Init loss function failed (%s)\n", STATUS_INIT_FAIL, param.StringValue("-loss").c_str());
				return STATUS_INIT_FAIL;
			}
			return STATUS_OK;
		}

		inline int InitDataSet(Params &param){
			string drt_type = param.StringValue("-drt");
			ToLowerCase(drt_type);
			if (drt_type == "online"){
				int buf_size = param.IntValue("-bs");
				int chunk_size = param.IntValue("-cs");
				this->pDataset = new OnlineDataSet<FeatType, LabelType>(param.IntValue("-passes"),
					param.BoolValue("-norm"), buf_size, chunk_size);

				if (this->pDataset == NULL){
					fprintf(stderr, "Error %d: init dataset failed! (%s)\n", STATUS_INIT_FAIL, drt_type.c_str());
					return STATUS_INIT_FAIL;
				}

				const string& mp_buf_type = param.StringValue("-mbt");
				int mp_buf_size = param.IntValue("-mbs");
				try{
					((OnlineDataSet<FeatType, LabelType>*)this->pDataset)->ConfigBuffer(buf_size, chunk_size, mp_buf_type, mp_buf_size);
				}
				catch (std::invalid_argument& ex){
					fprintf(stderr, "%s\n", ex.what());
					return STATUS_INVALID_ARGUMENT;
				}
			}
			else if (drt_type == "batch"){
				fprintf(stderr, "Error %d: batch dataset is not supported yet\n", STATUS_INVALID_ARGUMENT);
				return STATUS_INVALID_ARGUMENT;
			}
			else{
				fprintf(stderr, "Error %d: Unrecognized data reader type (%s)\n", STATUS_INVALID_ARGUMENT, drt_type.c_str());
				return STATUS_INVALID_ARGUMENT;
			}


			return STATUS_OK;;
		}

		inline int InitModel(Params &param){
			//check model type
			string model = param.StringValue("-m");
			this->pModel = (LearnModel<FeatType, LabelType>*)Registry::CreateObject(model, this->pLossFunc);
			if (this->pModel == NULL){
				fprintf(stderr, "Error %d: init online model failed! (%s)\n", STATUS_INIT_FAIL, model.c_str());
				return STATUS_INIT_FAIL;
			}

			const string& modelType = this->pModel->GetModelType();
			if (modelType == "online"){
				this->pOnlineModel = (OnlineModel<FeatType, LabelType>*)this->pModel;
				if (this->pOnlineModel == NULL){
				}
			}
			else{
				fprintf(stderr, "Error: unsupported model type %s!\n", modelType.c_str());
				return STATUS_INVALID_ARGUMENT;
			}

			try{
				this->pModel->SetParameter(param);
			}
			catch (invalid_argument &ex){
				fprintf(stderr, "%s\n", ex.what());
				return STATUS_INVALID_ARGUMENT;
			}

			return STATUS_OK;
		}

		inline int InitOptimizer(Params &param){
			string optType = param.StringValue("-opt");
			ToLowerCase(optType);
			this->pOpti = (Optimizer<FeatType, LabelType>*)
				Registry::CreateObject(optType, this->pOnlineModel, this->pDataset);

			if (this->pOpti == NULL) {
				fprintf(stderr, "Error %d: init optimizer failed! (%s)\n", STATUS_INIT_FAIL, optType.c_str());
				return STATUS_INIT_FAIL;
			}
			return STATUS_OK;
		}

	public:
		int Initialize(Params &param){
			this->Release();
			int errCode = STATUS_OK;
			errCode = this->InitDataSet(param);
			if (errCode == STATUS_OK){
				errCode = this->InitLoss(param);
			}
			if (errCode == STATUS_OK){
				errCode = this->InitModel(param);
			}
			if (errCode == STATUS_OK){
				errCode = this->InitOptimizer(param);
			}
			if (errCode != STATUS_OK){
				return errCode;
			}
			this->pParam = &param;
			return errCode;
		}

		int Run(){
			if (this->pDataset->Load(this->pParam->StringValue("-i"), this->pParam->StringValue("-c"),
				this->pParam->StringValue("-df")) == false){
				fprintf(stderr, "Error %d: Load dataset failed\n", STATUS_IO_ERROR);
				return STATUS_IO_ERROR;
			}

			this->pModel->PrintModelSettings();
			//learning the model
			double time1 = get_current_time();

			float l_errRate = this->pOpti->Train();

			if (this->pParam->StringValue("-om").length() > 0){
				this->pModel->SaveModel(this->pParam->StringValue("-om"));
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
				OnlineDataSet<FeatType, LabelType> testset(1, this->pParam->BoolValue("-norm"),
					this->pParam->IntValue("-bs"), this->pParam->IntValue("-cs"));
				if (testset.Load(this->pParam->StringValue("-t"), this->pParam->StringValue("-tc"),
					this->pParam->StringValue("-df")) == true) {
					float t_errRate(0);	//test error rate
					t_errRate = this->pOpti->Test(testset);
					time3 = get_current_time();

					printf("Test error rate: %.2f %%\n", t_errRate * 100);
					printf("Test time: %.3f s\n", (float)(time3 - time2));
				}
				else{
					fprintf(stderr, "load test set failed!\n");
				}
			}

			return STATUS_OK;
		}

		public:
			void InitParams(Params& param){
				string overview = "Sparse Online Learning Library";
				string syntax = "SOL [options] -i train_file";
				string example = "SOL -i train_file -m SGD";

				param.Init(overview, syntax, example);

				param.add_option("", 0, 1, "help message (model, optimizer, loss, io)", "-help", " ");

				//input & output
				param.add_option("", 0, 1, "training file", "-i", "Input Output");
				param.add_option("", 0, 1, "test file", "-t", "Input Output");
				param.add_option("", 0, 1, "cached training file", "-c", "Input Output");
				param.add_option("", 0, 1, "cached test file", "-tc", "Input Output");

				param.add_option(init_data_format, 0, 1, "Dataset Format", "-df", "Input Output");
				param.add_option(init_data_reader_type, 0, 1, "data reader type (online or batch)", "-drt", "Input Output");
				param.add_option(init_buf_size, 0, 1, "Buffer Size: number of chunks for buffering", "-bs", "Input Output");
				param.add_option(init_chunk_size, 0, 1, "Chunk Size: number of examples in a chunk", "-cs", "Input Output");
				param.add_option(init_normalize, 0, 0, "whether normalize the data", "-norm", "Input Output");

				//Training Settings
				param.add_option("", false, 1, "input existing model", "-im", "Training Settings");
				param.add_option("", false, 1, "output readable model", "-om", "Training Settings");
				param.add_option(1, 0, 1, "number of passes", "-passes", "Training Settings");
				param.add_option(init_mp_buf_type, 0, 1, "Multipass Buffer Type", "-mbt", "Training Settings");
				param.add_option(init_mp_buf_size, 0, 1, "Multipass Buffer Size", "-mbs", "Training Settings");

				//loss function
				param.add_option(init_loss_type, 0, 1, "loss function type", "-loss", "Loss Functions");

				//model setting
				param.add_option(init_algo_method, 0, 1, "learning model:", "-m", "Model Settings");
				param.add_option(init_eta, 0, 1, "learning rate", "-eta", "Model Settings");
				param.add_option(init_power_t, 0, 1, "power t of decaying learning rate", "-power_t", "Model Settings");
				param.add_option(init_initial_t, 0, 1, "initial iteration number", "-t0", "Model Settings");
				param.add_option(init_lambda, 0, 1, "l1 regularization", "-l1", "Model Settings");
				param.add_option(init_k, 0, 1,
					"number of k in truncated gradient descent or feature selection", "-k", "Model Settings");
				param.add_option(init_gammarou, 0, 1, "gamma times rou in enhanced RDA (RDA_E)", "-grou", "Model Settings");
				param.add_option(init_delta, 0, 1, "delta in Adaptive algorithms(Ada-)", "-delta", "Model Settings");
				param.add_option(init_r, 0, 1, "r in Confidence weighted algorithms and SOSOL", "-r", "Model Settings");

				//optimizer
				param.add_option(init_opt_type, 0, 1, "optimization algorithm", "-opt", "Optimizer");
			}
	};


}

#endif

