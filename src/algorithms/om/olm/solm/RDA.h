/************************************************************************
> File Name: RDA_L1.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 8/19/2013 Monday 1:49:43 PM
> Functions: Enhanced L1-RDA method
> Reference:
Xiao L. Dual averaging methods for regularized stochastic learning
and online optimization[J]. The Journal of Machine Learning Research,
2010, 9999: 2543-2596.
************************************************************************/

#ifndef HEADER_RDA_L1
#define HEADER_RDA_L1

#include "SparseOnlineLinearModel.h"

/**
*  namespace: Batch and Online Classification
*/
namespace BOC {
	template <typename FeatType, typename LabelType>
	class RDA : public SparseOnlineLinearModel < FeatType, LabelType > {

		DECLARE_CLASS

	protected:
		float gamma_rou;
		s_array<float> gtVec; //average gradient vector

		/**
		 * @Synopsis Constructors
		 */
	public:
		RDA(LossFunction<FeatType, LabelType> *lossFunc) :
			SparseOnlineLinearModel<FeatType, LabelType>(lossFunc) {
			this->modelName = "RDA";
			this->gamma_rou = 0;
			this->gtVec.resize(this->weightDim);
			//initail_t should be no less than 1,for the safety of update at the first step
			this->initial_t = 1;
		}

		~RDA(){
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
			printf("gamma rou: %g\n", this->gamma_rou);
		}

		/**
		 * @Synopsis SetParameter set parameters for the learning model
		 *
		 * @Param param
		 */
		virtual void SetParameter(BOC::Params &param){
			OnlineLinearModel<FeatType, LabelType>::SetParameter(param);
			this->gamma_rou = param.FloatValue("-grou");
			INVALID_ARGUMENT_EXCEPTION(gamma_rou, this->gamma_rou >= 0, "no smaller than 0");
		}

		/**
		 * @Synopsis BeginTrain Reset the optimizer to the initialization status of training
		 */
		virtual void BeginTrain() {
			SparseOnlineLinearModel<FeatType, LabelType>::BeginTrain();

			this->gtVec.zeros();
			//initail_t should be no less than 1,for the safety of update at the first step
			if (this->initial_t < 1)
				this->initial_t = 1;
		}

		/**
		 * @Synopsis Iterate Iteration of online learning
		 *
		 * @Param x current input data example
		 *
		 * @Returns  prediction of the current example
		 */
		virtual float Iterate(const DataPoint<FeatType, LabelType> &x) {
			this->curIterNum += 1;
			float eta_coeff_time = pEta_sqrt(this->curIterNum, this->power_t);
			this->eta = this->eta0 / eta_coeff_time;

			size_t featDim = x.indexes.size();
			IndexType index_i = 0;
			//obtain w_t
			float lambda_t = this->lambda * this->curIterNum + this->gamma_rou * eta_coeff_time;

			for (size_t i = 0; i < featDim; i++) {
				index_i = x.indexes[i];
				this->weightVec[index_i] = -this->eta *
					trunc_weight(this->gtVec[index_i], lambda_t);
			}
			//bias
			this->weightVec[0] = -this->eta * this->gtVec[0];

			//predict
			float y = this->TrainPredict(this->weightVec, x);
			float gt_i = this->lossFunc->GetGradient(x.label, y);

			//update the coeffs
			for (size_t i = 0; i < featDim; i++)
				this->gtVec[x.indexes[i]] += gt_i * x.features[i];
			//bias term
			this->gtVec[0] += gt_i;

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
				this->gtVec.reserve(new_dim + 1);
				this->gtVec.resize(new_dim + 1);
				this->gtVec.zeros(this->gtVec.begin + this->weightDim,
					this->gtVec.end);
				SparseOnlineLinearModel<FeatType, LabelType>::UpdateModelDimention(new_dim);
			}
		}

	};

	IMPLEMENT_MODEL_CLASS(RDA, "Regularized Dual Averaging")
}

#endif
