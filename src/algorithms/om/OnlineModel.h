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

#include <vector>

/**
*  namespace: Batch and Online Classification
*/
namespace BOC {

	template <typename FeatType, typename LabelType>
	class OnlineModel : public LearnModel < FeatType, LabelType > {
#pragma region Class Members
	protected:
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
		OnlineModel(LossFunction<FeatType, LabelType> *lossFunc, int classNum)
			: LearnModel<FeatType, LabelType>(lossFunc, classNum),
			power_t(0), eta0(0), eta(0),
			curIterNum(0), initial_t(0) {
			this->modelType = "online";
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
		 * @Synopsis IterateBC Iteration of online learning for binary classification
		 *
		 * @Param x current input data example
		 *
		 * @Returns  predicted class of the current example
		 */
		virtual int IterateBC(const DataPoint<FeatType, LabelType> &x, float* predict) = 0;

		/**
		 * @Synopsis IterateMC Iteration of online learning for multiclass classification
		 *
		 * @Param x current input data example
		 *
		 * @Returns  predicted class of the current example
		 */
		virtual int IterateMC(const DataPoint<FeatType, LabelType> &x, float* predict) = 0;

#pragma endregion Train Related
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

