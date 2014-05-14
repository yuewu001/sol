/*************************************************************************
  > File Name: SGD.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: 8/19/2013 Monday 10:37:08 AM
  > Functions: Stochastic Gradient Descent
 ************************************************************************/

#ifndef HEADER_SOLM_SGD
#define HEADER_SOLM_SGD

#include "SparseOnlineLinearModel.h"

/**
*  namespace: Batch and Online Classification
*/
namespace BOC {
	template <typename FeatType, typename LabelType>
	class SGD: public SparseOnlineLinearModel<FeatType, LabelType> {
		DECLARE_CLASS
	protected:
		float(*pEta_time)(size_t t, float pt);

		/**
		 * @Synopsis Constructors
		 */
	public:
		SGD(LossFunction<FeatType, LabelType> *lossFunc) :
			SparseOnlineLinearModel<FeatType, LabelType>(lossFunc) {
				this->id_str = SGD::classInfo.GetType();
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
			SparseOnlineLinearModel<FeatType, LabelType>::BeginTrain();

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
		 * @Synopsis Iterate Iteration of online learning
		 *
		 * @Param x current input data example
		 *
		 * @Returns  prediction of the current example
		 */
		virtual float Iterate(const DataPoint<FeatType, LabelType> &x) {
			this->curIterNum++;
			this->eta = this->eta0 / this->pEta_time(this->curIterNum, this->power_t);

			float y = this->Predict(x);
			size_t featDim = x.indexes.size();
			float gt_i = this->lossFunc->GetGradient(x.label, y);

			for (size_t i = 0; i < featDim; i++)
				this->weightVec[x.indexes[i]] -= this->eta * gt_i * x.features[i];

			//update bias 
			this->weightVec[0] -= this->eta * gt_i;
			return y;
		}
	};

	IMPLEMENT_CLASS(SGD)
}
#endif