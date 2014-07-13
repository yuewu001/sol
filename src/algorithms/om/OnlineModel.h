/*************************************************************************
	> File Name: OnlineModel.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 5/5/2014 3:24:43 PM
	> Functions: Interface for online model
	************************************************************************/
#ifndef HEADER_ONLINE_MODEL
#define HEADER_ONLINE_MODEL

#include "../LearnModel.h"
#include "../../utils/util.h"

/**
*  namespace: Batch and Online Classification
*/
namespace BOC {
	template <typename FeatType, typename LabelType>
	class OnlineModel : public LearnModel < FeatType, LabelType > {
#pragma region Class Members
	protected:
        //number of classes
		int class_num;
        //number of weight vectors
		int classfier_num;
		//power_t of the decreasing coefficient of learning rate
		float power_t;
		//initial learning rate
		float eta0;
		//learning rate
		float eta;

		//iteration number
		size_t curIterNum;

		//initial learning step
		size_t initial_t;

	public:
		/**
		 * @Synopsis Iterate Iteration of online learning 
		 *
		 * @Param x current input data example
		 *
		 * @Returns  predicted class of the current example
		 */

#pragma endregion Class Members

#pragma region Constructors and Basic Functions
	public:
		OnlineModel(LossFunction<FeatType, LabelType> *lossFunc, int classNum = 2)
			: LearnModel<FeatType, LabelType>(lossFunc),
			class_num(2), classfier_num(1), power_t(0), eta0(0), eta(0),
			curIterNum(0), initial_t(0) {
			this->modelType = "online";
			this->class_num = classNum;
			this->classfier_num = this->class_num == 2 ? 1 : this->class_num;
			INVALID_ARGUMENT_EXCEPTION(class_num, class_num > 1, "no smaller than 2");
		}

		virtual ~OnlineModel() {
		}

		/**
		 * PrintModelSettings print the info of optimization algorithm
		 */
		virtual void PrintModelSettings() const {
			LearnModel<FeatType, LabelType>::PrintModelSettings();

			printf("Online Learning\n");
			printf("\tLearning Rate: %g\n", this->eta0);
			printf("\tInitial t  : %lu\n", this->initial_t);
			printf("\tPower t : %g\n", this->power_t);
		}

		/**
		*  GetClassfierNum Get the number of classifiers
		*/
		int GetClassfierNum() const { return this->classfier_num; }

#pragma endregion Constructors and Basic Functions

#pragma region Train Related
	public:
		/**
		 * @Synopsis BeginTrain Reset the optimizer to the initialization status of training
		 */
		virtual void BeginTrain() {
			this->curIterNum = this->initial_t;
		}

		/**
		 * @Synopsis EndTrain called when a train ends
		 */
		virtual void EndTrain(){}

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
		virtual void SetParameter(BOC::Params &param){
			this->initial_t = param.IntValue("-t0");
			INVALID_ARGUMENT_EXCEPTION(initial_t, this->initial_t >= 0, "no smaller than 0");
			this->power_t = param.FloatValue("-power_t");
			INVALID_ARGUMENT_EXCEPTION(power_t, this->power_t >= 0, "no smaller than 0");
			this->eta0 = param.FloatValue("-eta");
			INVALID_ARGUMENT_EXCEPTION(eta0, this->eta0 >= 0, "no smaller than 0");
		}

		/**
		 * @Synopsis Iterate Iteration of online learning
		 *
		 * @Param x current input data example
		 *
		 * @Returns  predicted class of the current example
		 */
		int IterateBCDelegate(const DataPoint<FeatType, LabelType> &x, float& predict){
			return this->IterateBC(x, predict);
		}

		/**
		 * @Synopsis IterateBC Iteration of online learning for binary classification
		 *
		 * @Param x current input data example
		 *
		 * @Returns  predicted class of the current example
		 */
		virtual int IterateBC(const DataPoint<FeatType, LabelType> &x, float& predict) = 0;

		/**
		 * @Synopsis IterateMC Iteration of online learning for multiclass classification
		 *
		 * @Param x current input data example
		 *
		 * @Returns  predicted class of the current example
		 */
		virtual int IterateMC(const DataPoint<FeatType, LabelType> &x, float& predict) = 0;

#pragma endregion Train Related

#pragma region Test related
    public:
		/**
		 * @Synopsis Test_Predict prediction function for test
		 *
		 * @Param data input data sample
		 * @Param predicts predicted values for each classifier
		 *
		 * @Returns predicted class
		 */
		virtual int Predict(const DataPoint<FeatType, LabelType> &data, vector<float> predicts){
			return this->PredictBC(data, predicts);
		}

	protected:
		/**
		 * @Synopsis PredictBC prediction function for test (binary classification)
		 *
		 * @Param data input data sample
		 * @Param predicts predicted values for each classifier
		 *
		 * @Returns predicted class
		 */
		int PredictBC(const DataPoint<FeatType, LabelType> &data, vector<float>& predicts) {
			predicts[0] = this->PredictBC(data);

			if (this->IsCorrect(data.label, predicts[0]) == false){
				return -data.label;
			}
			else{
				return data.label;
			}
		}

		/**
		 * @Synopsis Predict prediction function for test (multiclass classification)
		 *
		 * @Param data input data sample
		 * @Param predicts predicted values for each classifier
		 *
		 * @Returns predicted class
		 */
		int PredictMC(const DataPoint<FeatType, LabelType> &data, vector<float>& predicts) {
			for (int k = 0; k < this->classfier_num; ++k){
				predicts[k] = this->Test_PredictMC(k, data);
			}

			return std::max_element(predicts.begin(), predicts.end()) - predicts.begin() + 1;
		}

		/**
		 * @Synopsis Test_PredictBC prediction function for test (binary classification)
		 *
		 * @Param data input data sample
		 *
		 * @Returns predicted value for the data 
		 */
		virtual float PredictBC(const DataPoint<FeatType, LabelType> &data) = 0;

		/**
		 * @Synopsis Test_PredictMC prediction function for test (multiclass classification)
		 *
         * @Param classId: specified classifer
		 * @Param data input data sample
		 *
		 * @Returns predicted value for the data on the specified classifier
		 */
		virtual float PredictMC(int classId, const DataPoint<FeatType, LabelType> &data) = 0;

#pragma endregion Test related
	};

#pragma region Learning Rate Functions
	//calculate learning rate
	inline float pEta_general(size_t t, float pt){
		return powf((float)t, pt);
	}
	inline float pEta_sqrt(size_t t, float pt){
		return sqrtf((float)t);
	}
	inline float pEta_linear(size_t t, float pt){
		return (float)t;
	}
	inline float pEta_const(size_t t, float pt){
		return 1;
	}
#pragma endregion Learning Rate Functions

}

#endif

