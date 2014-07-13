/*************************************************************************
> File Name: Ada_RDA.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/18 Sunday 17:25:54
> Functions: Adaptive RDA
> Reference:
Duchi J, Hazan E, Singer Y. Adaptive subgradient methods for
online learning and stochastic optimization[J]. The Journal of
Machine Learning Research, 2011, 999999: 2121-2159.

This file implements the L1 regularization
************************************************************************/

#ifndef HEADER_ADA_RDA
#define HEADER_ADA_RDA

#include "SparseOnlineLinearModel.h"

/**
*  namespace: Batch and Online Classification
*/
namespace BOC {
	template <typename FeatType, typename LabelType>
	class Ada_RDA : public SparseOnlineLinearModel < FeatType, LabelType > {

		DECLARE_CLASS

	protected:
		float delta;
		s_array<float> s;
		s_array<float> u_t;
	public:
		Ada_RDA(LossFunction<FeatType, LabelType> *lossFunc) :
			SparseOnlineLinearModel<FeatType, LabelType>(lossFunc){
			this->modelName = "Ada-RDA";
			this->delta = 0;;
			this->s.resize(this->weightDim);
			this->u_t.resize(this->weightDim);
		}

		virtual ~Ada_RDA() {
		}

		/**
		 * @Synopsis inherited functions
		 */
	public:
		/**
		 * PrintModelSettings print the info of optimization algorithm
		 */
		virtual void PrintModelSettings() const {
			SparseOnlineLinearModel<FeatType, LabelType>::PrintModelSettings();
			printf("\tdelta : %g\n", this->delta);
		}

		/**
		 * @Synopsis SetParameter set parameters for the learning model
		 *
		 * @Param param
		 */
		virtual void SetParameter(BOC::Params &param){
			SparseOnlineLinearModel<FeatType, LabelType>::SetParameter(param);
			this->delta = param.FloatValue("-delta");
			INVALID_ARGUMENT_EXCEPTION(delta, this->delta >= 0, "no smaller than 0");
		}

		/**
		 * @Synopsis BeginTrain Reset the optimizer to the initialization status of training
		 */
		virtual void BeginTrain() {
			SparseOnlineLinearModel<FeatType, LabelType>::BeginTrain();

			//reset time stamp
			this->s.zeros();
			this->u_t.zeros();
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
			size_t featDim = x.indexes.size();
			IndexType index_i = 0;

			//obtain w_t
			for (size_t i = 0; i < featDim; i++) {
				index_i = x.indexes[i];
				//lazy update
				//update s[i]
				float Htii = this->delta + sqrtf(s[index_i]);
				this->weightVec[index_i] = -this->eta0 / Htii *
					trunc_weight(u_t[index_i], this->lambda * (this->curIterNum - 1));
			}

			//predict 
			float y = this->TrainPredict(this->weightVec,x);
			//get gradient
			float gt = this->lossFunc->GetGradient(x.label, y);
			if (gt != 0){
				float gt_i = 0;
				//update
				for (size_t i = 0; i < featDim; i++) {
					index_i = x.indexes[i];
					gt_i = gt * x.features[i];

					this->s[index_i] += gt_i * gt_i;
					this->u_t[index_i] += gt_i;
				}
				//bias term
				this->s[0] += gt * gt;
				this->u_t[0] += gt;
				float Htii = this->delta + sqrtf(s[0]);
				this->weightVec[0] = -u_t[0] * this->eta0 / Htii;
			}
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
				this->s.reserve(new_dim + 1);
				this->s.resize(new_dim + 1);
				//set the rest to zero
				this->s.zeros(this->s.begin + this->weightDim,
					this->s.end);

				this->u_t.reserve(new_dim + 1);
				this->u_t.resize(new_dim + 1);
				//set the rest to zero
				this->u_t.zeros(this->u_t.begin + this->weightDim,
					this->u_t.end);

				SparseOnlineLinearModel<FeatType, LabelType>::UpdateModelDimention(new_dim);
			}
		}
	};

	IMPLEMENT_MODEL_CLASS(Ada_RDA, "Adaptive RDA")
}
#endif
