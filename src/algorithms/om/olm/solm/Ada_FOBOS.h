/*************************************************************************
> File Name: Ada_FOBOS.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: Sat 26 Oct 2013 12:17:04 PM SGT
> Descriptions: adaptive fobos algorithm
> Reference:
Duchi J, Hazan E, Singer Y. Adaptive subgradient methods for
online learning and stochastic optimization[J]. The Journal of
Machine Learning Research, 2011, 999999: 2121-2159.

This file implements the L1 regularization
************************************************************************/
#ifndef HEADER_ADA_FOBOS
#define HEADER_ADA_FOBOS

#include "SparseOnlineLinearModel.h"

/**
*  namespace: Batch and Online Classification
*/
namespace BOC {
	template <typename FeatType, typename LabelType>
	class Ada_FOBOS : public SparseOnlineLinearModel < FeatType, LabelType > {

		DECLARE_CLASS

	protected:
		float delta;
		s_array<size_t> timeStamp;
		s_array<float> s;
		s_array<float> u_t;

		/**
		 * @Synopsis Constructors
		 */
	public:
		Ada_FOBOS(LossFunction<FeatType, LabelType> *lossFunc, int classNum) :
			SparseOnlineLinearModel<FeatType, LabelType>(lossFunc, classNum){
			this->modelName = "Ada-FOBOS";
			this->delta = init_delta;
			this->timeStamp.resize(this->weightDim);
			this->s.resize(this->weightDim);
			this->u_t.resize(this->weightDim);
		}

		virtual ~Ada_FOBOS() {
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
			float val = param.FloatValue("-delta");
			if (val >= 0){
				this->delta = val;
			}
		}

		/**
		 * @Synopsis BeginTrain Reset the optimizer to the initialization status of training
		 */
		virtual void BeginTrain() {
			INVALID_ARGUMENT_EXCEPTION(delta, this->delta >= 0, "no smaller than 0");
			SparseOnlineLinearModel<FeatType, LabelType>::BeginTrain();

			//reset time stamp
			this->timeStamp.zeros();
			this->s.zeros();
			this->u_t.zeros();
		}

		/**
		 * @Synopsis EndTrain called when a train ends
		 */
		virtual void EndTrain() {
			size_t iterNum = this->curIterNum + 1;
			float alpha = 0;
			for (IndexType index_i = 1; index_i < this->weightDim; index_i++) {
				//update s[i]
				float Ht0i = this->delta + s[index_i];
				alpha = this->lambda * this->eta0 * (iterNum - this->timeStamp[index_i]) / Ht0i;
				this->weightVec[index_i] = trunc_weight(this->weightVec[index_i], alpha);
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
			float y = this->TrainPredict(this->weightVec, x);

			//get gradient
			float gt = 0;
			this->lossFunc->GetGradient(x.label, &y, &gt);
			if (gt != 0){
				float gt_i = 0;
				IndexType index_i = 0;
				float alpha = this->eta0 * this->lambda;
				size_t featDim = x.indexes.size();

				//update s[i]
				for (size_t i = 0; i < featDim; i++) {
					index_i = x.indexes[i];
					gt_i = gt * x.features[i];

					float Ht0i = this->delta + s[index_i];
					this->s[index_i] = sqrt(s[index_i] * s[index_i] + gt_i * gt_i);
					float Htii = this->delta + s[index_i];
					//obtain w_(t + 1),i
					this->weightVec[index_i] -= this->eta0 * gt_i / Htii;

					//to obtain w_(t + 1),i, first calculate w_t,i
					this->weightVec[index_i] = trunc_weight(this->weightVec[index_i],
						alpha * (this->curIterNum - this->timeStamp[index_i]) / Ht0i);

					//update the time stamp
					this->timeStamp[index_i] = this->curIterNum;
				}

				//bias term
				this->s[0] = sqrt(s[0] * s[0] + gt * gt);
				float Htii = this->delta + s[0];
				this->weightVec[0] -= this->eta0 * gt / Htii;
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
				this->timeStamp.resize(new_dim + 1);
				//set the rest to zero
				this->timeStamp.zeros(this->timeStamp.begin + this->weightDim,
					this->timeStamp.end);

				this->s.resize(new_dim + 1);
				//set the rest to zero
				this->s.zeros(this->s.begin + this->weightDim,
					this->s.end);

				this->u_t.resize(new_dim + 1);
				//set the rest to zero
				this->u_t.zeros(this->u_t.begin + this->weightDim,
					this->u_t.end);

				SparseOnlineLinearModel<FeatType, LabelType>::UpdateModelDimention(new_dim);
			}
		}
	};

	IMPLEMENT_MODEL_CLASS(Ada_FOBOS, "Adaptive FOBOS")
}

#endif
