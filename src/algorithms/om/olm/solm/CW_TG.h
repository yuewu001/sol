/*************************************************************************
> File Name: Diagonal AROW
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/18 Sunday 17:25:54
> Functions: Diagonal Adaptive Regularization of Weight Vectors
> Reference:
Crammer, Koby, Alex Kulesza, and Mark Dredze. "Adaptive regularization
of weight vectors." Machine Learning (2009): 1-33.
************************************************************************/

#ifndef HEADER_CW_TG
#define HEADER_CW_TG

#include "SparseOnlineLinearModel.h"
#include "../../../../loss/SquaredHingeLoss.h"

/**
*  namespace: Batch and Online Classification
*/
namespace BOC {
	template <typename FeatType, typename LabelType>
	class CW_TG : public SparseOnlineLinearModel < FeatType, LabelType > {

		DECLARE_CLASS

	protected:
		float r;
		s_array<float> sigma_w;
		s_array<size_t> timeStamp;
		s_array<float> sum_rate;

		size_t iter_num;

	public:
		CW_TG(LossFunction<FeatType, LabelType> *lossFunc) :
			SparseOnlineLinearModel<FeatType, LabelType>(lossFunc){
			this->modelName = "Cw-TG";
			this->r = 0;
			this->sigma_w.resize(this->weightDim);
			this->timeStamp.resize(this->weightDim);
			this->lossFunc = new SquaredHingeLoss < FeatType, LabelType > ;
		}
		virtual ~CW_TG(){
			if (this->lossFunc != NULL){
				delete this->lossFunc;
				this->lossFunc = NULL;
			}
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
			printf("\tr:\t%g\n", this->r);
		}

		/**
		 * @Synopsis SetParameter set parameters for the learning model
		 *
		 * @Param param
		 */
		virtual void SetParameter(BOC::Params &param){
			SparseOnlineLinearModel<FeatType, LabelType>::SetParameter(param);
			this->r = param.FloatValue("-r");
			INVALID_ARGUMENT_EXCEPTION(r, this->r >= 0, "no smaller than 0");
		}

		/**
		 * @Synopsis BeginTrain Reset the optimizer to the initialization status of training
		 */
		virtual void BeginTrain() {
			SparseOnlineLinearModel<FeatType, LabelType>::BeginTrain();

			this->timeStamp.zeros();
			this->sigma_w.set_value(1);
			this->sum_rate.push_back(0);
			this->iter_num = 1; //force to begin from 1, as sum_rate depends on this value
		}

		/**
		 * @Synopsis EndTrain called when a train ends
		 */
		virtual void EndTrain() {
			float gravity = 0;
			//this->beta_t = 1.f / this->r;
			for (IndexType index_i = 1; index_i < this->weightDim; index_i++) {
				//L1 lazy update
				gravity = this->sum_rate.last() - this->sum_rate[this->timeStamp[index_i]];
				//size_t stepK = this->curIterNum - this->timeStamp[index_i];
				//gravity = stepK * this->lambda * this->beta_t / 2.f;

				//this->timeStamp[index_i] = this->curIterNum;
				this->weightVec[index_i] = trunc_weight(this->weightVec[index_i],
					gravity *(this->sigma_w[index_i]));
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
			IndexType* p_index = x.indexes.begin;
			float* p_feat = x.features.begin;

			//obtain w_t
			float y = this->Predict(x);
			float gt_i = this->lossFunc->GetGradient(x.label, y);
			//update w_t
			if (gt_i != 0){
				//calculate learning rate
				this->eta = this->r;
				while (p_index != x.indexes.end){
					this->eta += (*p_feat) * (*p_feat) * this->sigma_w[*p_index];
					p_index++; p_feat++;
				}

				this->eta = 0.5f / this->eta;
				this->sum_rate.push_back(this->sum_rate.last() +
					this->eta * this->lambda);
				gt_i *= this->eta;

				float last_g_sum = this->sum_rate.last();
				p_index = x.indexes.begin;
				p_feat = x.features.begin;
				while (p_index != x.indexes.end){
					//update u_t
					this->weightVec[*p_index] -= gt_i *
						(*p_feat) * this->sigma_w[*p_index];

					//L1 lazy update
					//size_t stepK = this->iter_num - this->timeStamp[*p_index];
					float gravity = last_g_sum -
						this->sum_rate[this->timeStamp[*p_index]];
					//float gravity = stepK * this->lambda * this->beta_t / 2.f;
					this->timeStamp[*p_index] = this->iter_num;

					this->weightVec[*p_index] =
						trunc_weight(this->weightVec[*p_index],
						gravity *(this->sigma_w[*p_index]));

					//update sigma_w
					this->sigma_w[*p_index] *= this->r / (this->r +
						this->sigma_w[*p_index] * (*p_feat) * (*p_feat));
					p_index++; p_feat++;
				}

				//bias term
				this->weightVec[0] -= gt_i * this->sigma_w[0];
				this->sigma_w[0] *= this->r / (this->r + this->sigma_w[0]);
				this->timeStamp[0] = this->iter_num;

				this->iter_num++;
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
				this->sigma_w.reserve(new_dim + 1);
				this->sigma_w.resize(new_dim + 1);
				//set the rest to one
				this->sigma_w.set_value(this->sigma_w.begin + this->weightDim,
					this->sigma_w.end, 1);

				this->timeStamp.reserve(new_dim + 1);
				this->timeStamp.resize(new_dim + 1);
				//set the rest to zero
				this->timeStamp.zeros(this->timeStamp.begin + this->weightDim,
					this->timeStamp.end);

				SparseOnlineLinearModel<FeatType, LabelType>::UpdateModelDimention(new_dim);
			}
		}
	};

	IMPLEMENT_MODEL_CLASS(CW_TG, "Confidence Weighted Truncated Gradient")
}
#endif
