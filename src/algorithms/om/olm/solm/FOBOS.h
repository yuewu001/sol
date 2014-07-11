/*************************************************************************
  > File Name: FOBOS.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: 2013/8/20 Tuesday 11:14:54
  > Functions: FOBOS: Efficient Online Batch Learning Using
  Forward Backward Splitting

  > Reference:
  Duchi J, Singer Y. Efficient online and batch learning using
  forward backward splitting[J]. The Journal of Machine Learning
  Research, 2009, 10: 2899-2934.

  This file implements the L1 and L2 square regularization
  ************************************************************************/

#ifndef HEADER_FOBOS
#define HEADER_FOBOS

#include "SparseOnlineLinearModel.h"

/**
 *  namespace: Batch and Online Classification
 */
namespace BOC {
	template <typename FeatType, typename LabelType>
	class FOBOS : public SparseOnlineLinearModel < FeatType, LabelType > {

		DECLARE_CLASS

	protected:
		s_array<size_t> timeStamp;
		float(*pEta_time)(size_t t, float pt);

	public:
		FOBOS(LossFunction<FeatType, LabelType> *lossFunc) :
			SparseOnlineLinearModel<FeatType, LabelType>(lossFunc){
			this->modelName = "FOBOS";
			this->timeStamp.resize(this->weightDim);
		}

		virtual ~FOBOS() {
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

			this->timeStamp.zeros();
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
		 * @Synopsis EndTrain called when a train ends
		 */
		virtual void EndTrain() {
			for (IndexType index_i = 1; index_i < this->weightDim; index_i++) {
				//truncated gradient
				size_t stepK = this->curIterNum - this->timeStamp[index_i];
				this->weightVec[index_i] = trunc_weight(this->weightVec[index_i],
					stepK * this->eta * this->lambda);
			}
			SparseOnlineLinearModel<FeatType, LabelType>::EndTrain();
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

			size_t featDim = x.indexes.size();
			float y = this->Predict(x);
			float gt_i = this->eta * this->lossFunc->GetGradient(x.label, y);

			IndexType index_i = 0;
			float alpha = this->eta * this->lambda;
			size_t stepK = 0;
			for (size_t i = 0; i < featDim; i++) {
				index_i = x.indexes[i];
				//update the weight
				this->weightVec[index_i] -= gt_i * x.features[i];

				//lazy update
				stepK = this->curIterNum - this->timeStamp[index_i];
				this->timeStamp[index_i] = this->curIterNum;

				this->weightVec[index_i] = trunc_weight(this->weightVec[index_i],
					stepK * alpha);
			}

			//update bias term
			this->weightVec[0] -= gt_i;

			return y;
		}

		/**
		 * @Synopsis UpdateModelDimention update dimension of the model,
		 * often caused by the increased dimension of data
		 *
		 * @Param new_dim new dimension
		 */
		virtual void UpdateModelDimention(IndexType new_dim) {
			if (new_dim < this->weightDim)
				return;
			else {
				this->timeStamp.reserve(new_dim + 1);
				this->timeStamp.resize(new_dim + 1);
				//set the rest to zero
				this->timeStamp.zeros(this->timeStamp.begin + this->weightDim,
					this->timeStamp.end);

				SparseOnlineLinearModel<FeatType, LabelType>::UpdateModelDimention(new_dim);
			}
		}
	};

	IMPLEMENT_MODEL_CLASS(FOBOS, "Forward Backward Splitting")
}

#endif
