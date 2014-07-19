/*************************************************************************
  > File Name: SGD.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: 8/19/2013 Monday 10:37:08 AM
  > Functions: Stochastic Gradient Descent
  ************************************************************************/

#ifndef HEADER_SOLM_SGD
#define HEADER_SOLM_SGD

#include "OnlineLinearModel.h"

/**
*  namespace: Batch and Online Classification
*/
namespace BOC {
	template <typename FeatType, typename LabelType>
	class SGD : public OnlineLinearModel < FeatType, LabelType > {

		DECLARE_CLASS

	protected:
		float(*pEta_time)(size_t t, float pt);

		/**
		 * @Synopsis Constructors
		 */
	public:
		SGD(LossFunction<FeatType, LabelType> *lossFunc, int classNum) :
			OnlineLinearModel<FeatType, LabelType>(lossFunc, classNum) {
			this->modelName = "SGD";
		}

		virtual ~SGD() {
		}

		/**
		 * @Synopsis inherited functions
		 */
	public:
		/**
		 * @Synopsis BeginTrain Reset the optimizer to the initialization status of training
		 */
		virtual void BeginTrain() {
			OnlineLinearModel<FeatType, LabelType>::BeginTrain();

			if (this->power_t == 0.5)
				this->pEta_time = pEta_sqrt;
			else if (this->power_t == 0)
				this->pEta_time = pEta_const;
			else if (this->power_t == 1)
				this->pEta_time = pEta_linear;
			else
				this->pEta_time = pEta_general;
		}

		/**
		 * @Synopsis UpdateWeightVec Update the weight vector
		 *
		 * @Param x current input data example
		 * @Param weightVec weight vector to be updated
         * @param gt common part of the gradient
		 *
		 */
		virtual void UpdateWeightVec(const DataPoint<FeatType, LabelType> &x, s_array<float>& weightVec, float gt){
			this->eta = this->eta0 / this->pEta_time(this->curIterNum, this->power_t);
			size_t featDim = x.indexes.size();

			for (size_t i = 0; i < featDim; i++){
				weightVec[x.indexes[i]] -= this->eta * gt * x.features[i];
			}

			//update bias 
			weightVec[0] -= this->eta * gt;
		}
	};

	IMPLEMENT_MODEL_CLASS(SGD, "Stochasitic Gradient Descent")
}
#endif
