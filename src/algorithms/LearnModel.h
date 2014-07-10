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

    //TO return the class infomation and create a new instance of the algorithm
#define IMPLEMENT_MODEL_CLASS(name, descr) \
	template <typename FeatType, typename LabelType> \
	ClassInfo name<FeatType, LabelType>::classInfo(#name, descr, name<FeatType, LabelType>::CreateObject); \
	\
	template <typename FeatType, typename LabelType> \
	void* name<FeatType, LabelType>::CreateObject(void *lossFunc, void* param2, void* param3) \
	{ return new name<FeatType, LabelType>((LossFunction<FeatType, LabelType>*)lossFunc); }



	template <typename FeatType, typename LabelType>
	//	class LearnModel : public Registry<FeatType, LabelType> {
	class LearnModel : public Registry {
	protected:
		//type of the model: online
		std::string modelType;
	public:
		const std::string& GetModelType() const { return modelType; }

	protected:
		LossFunction<FeatType, LabelType> *lossFunc;

	public:
		LearnModel(LossFunction<FeatType, LabelType> *lossFunc){
			this->lossFunc = lossFunc;
		}

		virtual ~LearnModel() {
		}

		/**
		 * PrintModelSettings print the info of optimization algorithm
		 */
		virtual void PrintModelSettings() const {
			printf("-----------------------------------------\n");
			printf("-----------------------------------------\n");
			printf("             Model Settings              \n");
			printf("-----------------------------------------\n");
			printf("-----------------------------------------\n");
		}

		/**
		 * PrintModelInfo print the info of trained model
		 */
		virtual void PrintModelInfo() const = 0;

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
		 * @Synopsis UpdateModelDimention update dimension of the model,
		 * often caused by the increased dimension of data
		 *
		 * @Param new_dim new dimension
		 */
		virtual void UpdateModelDimention(IndexType new_dim) = 0;

		/**
		 * @Synopsis SetParameter set the basic online learning parameters
		 *
		 */
		virtual void SetParameter(BOC::Params &param) = 0;

		/**
		 * @Synopsis Test_Predict prediction function for test
		 *
		 * @Param data input data sample
		 *
		 * @Returns predicted value
		 */
		virtual float Test_Predict(const DataPoint<FeatType, LabelType> &data) = 0;

		/**
		 * @Synopsis Predict prediction function for training
		 *
		 * @Param data input data sample
		 *
		 * @Returns predicted value
		 */
		virtual float Predict(const DataPoint<FeatType, LabelType> &data) = 0;

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

		/**
		 * @Synopsis common functions
		 */
	public:
		/**
		 * @Synopsis IsCorrect Judge if the predict is correct
		 *
		 * @Param label true label
		 * @Param predict predicted label
		 *
		 * @Returns true if correctly predicted
		 */
		virtual inline bool IsCorrect(LabelType label, float predict) {
			return this->lossFunc->IsCorrect(label, predict);
		}
	};
}

#endif
