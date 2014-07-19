/*************************************************************************
	> File Name: Model.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 5/5/2014 3:29:56 PM
	> Functions: Interface for learning model
	************************************************************************/

#ifndef HEADER_LEARN_MODEL
#define HEADER_LEARN_MODEL

#include "../loss/LossFunction.h"
#include "../utils/config.h"
#include "../utils/util.h"
#include "../utils/reflector.h"
#include "../utils/Params.h"
#include "../io/DataPoint.h"

#include <stdexcept>
#include <string>
#include <map>
#include <stdio.h>
using std::string;

/**
*  namespace: Batch and Online Classification
*/
namespace BOC {

#pragma region Macros for Reflector 

    //TO return the class infomation and create a new instance of the algorithm
#define IMPLEMENT_MODEL_CLASS(name, descr) \
	template <typename FeatType, typename LabelType> \
	ClassInfo name<FeatType, LabelType>::classInfo(#name, descr, name<FeatType, LabelType>::CreateObject); \
	\
	template <typename FeatType, typename LabelType> \
	void* name<FeatType, LabelType>::CreateObject(void *lossFunc, void* param2, void* param3) \
	    { return new name<FeatType, LabelType>((LossFunction<FeatType, LabelType>*)lossFunc, int(param2)); }

#pragma endregion Macros for Reflector 

	template <typename FeatType, typename LabelType>
	//	class LearnModel : public Registry<FeatType, LabelType> {
	class LearnModel : public Registry {
#pragma region Class Members
	protected:
		//number of classes
		int class_num;
        //number of weight vectors
		int classfier_num;
		//name of the model
		std::string modelName;
		//type of the model: online
		std::string modelType;
	public:
		const std::string& GetModelType() const { return modelType; }

	protected:
		LossFunction<FeatType, LabelType> *lossFunc;
#pragma endregion Class Members

#pragma region Constructors and Basic Functions
	public:
		LearnModel(LossFunction<FeatType, LabelType> *lossFunc, int classNum):
			lossFunc(lossFunc), class_num(classNum){
			this->classfier_num = this->class_num == 2 ? 1 : this->class_num;
			INVALID_ARGUMENT_EXCEPTION(class_num, class_num > 1, "no smaller than 2");

            //check if the loss function is approariate
			if (this->classfier_num == 1){
				if (this->lossFunc->GetLossType() != BC_LOSS_TYPE){
					throw invalid_argument("type of loss function (mc) is not consistent with the task (bc)");
				}
			}
			else{
				if (this->lossFunc->GetLossType() != MC_LOSS_TYPE){
					throw invalid_argument("type of loss function (bc) is not consistent with the task (mc)");
				}
			}
		}

		virtual ~LearnModel() {
		}

		/**
		 * PrintModelSettings print the info of optimization algorithm
		 */
		virtual void PrintModelSettings() const {
			printf("-----------------------------------------\n");
			printf("             Algorithm:\t%s\n", this->modelName.c_str());
			printf("-----------------------------------------\n");
			printf("             Model Settings              \n");
			printf("-----------------------------------------\n");
		}

		/**
		 * PrintModelInfo print the info of trained model
		 */
		virtual void PrintModelInfo() const = 0;

		/**
		*  GetClassfierNum Get the number of classifiers
		*/
		int GetClassfierNum() const { return this->classfier_num; }

#pragma endregion Constructors and Basic Functions

#pragma region IO related
	public:
		/**
		 * @Synopsis SaveModel save model to disk
		 *
		 * @Param filename  name to the saved file
		 *
		 * @Returns true if saved successfully
		 */
		virtual bool SaveModel(const string& filename) = 0;

		/**
		 * @Synopsis LoadModel load model from disk
		 *
		 * @Param filename path name of the model on disk
		 *
		 * @Returns true if load successfully
		 */
		virtual bool LoadModel(const string& filename) = 0;

	protected:
		/**
		 * @Synopsis SaveModelConfig save configuration of model to disk
		 *
		 * @Param os ostream object to which config are saved
		 *
		 * @Returns true if saved successfully
		 */
		virtual bool SaveModelConfig(std::ofstream &os) = 0;

		/**
		 * @Synopsis LoadModelConfig load configuration of model from disk
		 *
		 * @Param is istream object from which config are loaded
		 *
		 * @Returns true if load successfully
		 */
		virtual bool LoadModelConfig(std::ifstream &is) = 0;

		/**
		 * @Synopsis  SaveModelValue save model value to disk
		 *
		 * @Param os ostream object to which values are saved
		 *
		 * @Returns true if saved successfully
		 */
		virtual bool SaveModelValue(std::ofstream &os) = 0;

		/**
		 * @Synopsis LoadModelConfig load values of model from disk
		 *
		 * @Param is istream object from which values are loaded
		 *
		 * @Returns true if load successfully
		 */
		virtual bool LoadModelValue(std::ifstream &is) = 0;
#pragma endregion IO related

#pragma region Common Functions for Train and Test
	protected:
		/**
		 * @Synopsis IsCorrect Judge if the predict is correct
		 *
		 * @Param label true label
		 * @Param predict predicted label
		 *
		 * @Returns true if correctly predicted
		 */
		virtual inline bool IsCorrect(LabelType label, float* predict) {
			if (this->classfier_num == 1){
				return LossFunction<FeatType, LabelType>::Sign(*predict) == label ? true : false;
			}
			else{
				for (int i = 0; i < this->classfier_num; ++i){
					if (predict[i] > predict[label]){
						return false;
					}
				}
				return true;
			}
		}

		/**
		* @Synopsis: Get the class label for classifier
		*
		* @Param x current input data example
		* @Param predict_label predicted label for x
		*
		* @Returns label of x, if predicted label is the same to label, return 1, otherwise return -1
		*/
		int GetClassLabel(const DataPoint<FeatType, LabelType>& x, int predict_label = 1){
			return x.label == predict_label ? 1 : -1;
		}
#pragma endregion Common Functions for Train and Test

#pragma region Train Related
	public:
		/**
		 * @Synopsis BeginTrain Reset the optimizer to the initialization status of training
		 */
		virtual void BeginTrain() = 0;

		/**
		 * @Synopsis EndTrain called when a train ends
		 */
		virtual void EndTrain() = 0;

		/**
		 * @Synopsis SetParameter set the basic online learning parameters
		 *
		 */
		virtual void SetParameter(BOC::Params &param) = 0;

#pragma endregion Train Related

#pragma region Test related
	public:
		/**
		 * @Synopsis Predict prediction function for test
		 *
		 * @Param data input data sample
		 * @Param predicts predicted values for each classifier
		 *
		 * @Returns predicted class
		 */
		virtual int Predict(const DataPoint<FeatType, LabelType> &data, float* predicts) = 0;
#pragma endregion Test related

	};
}

#endif
