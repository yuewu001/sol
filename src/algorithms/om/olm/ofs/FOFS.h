/*************************************************************************
  > File Name: FOFS.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Tue 17 Dec 2013 04:59:06 AM EST
  > Descriptions: Online Feature Selection
  > Reference: Online Feature Selection and its applications
  ************************************************************************/
#ifndef HEADER_OPTIMISER_FOFS
#define HEADER_OPTIMISER_FOFS

#include "OnlineFeatureSelection.h"
#include "../../../../utils/MinHeap.h"

/**
*  namespace: Batch and Online Classification
*/
namespace BOC {
	template <typename FeatType, typename LabelType>
	class FOFS : public OnlineFeatureSelection < FeatType, LabelType > {

		DECLARE_CLASS

	protected:
		float w_norm;
		float norm_coeff;

		float shrinkage;
		float delta;
		s_array<float> abs_weightVec;

		MinHeap<float> minHeap;

	public:
		FOFS(LossFunction<FeatType, LabelType> *lossFunc) :
			OnlineFeatureSelection<FeatType, LabelType>(lossFunc) {
			this->modelName = "FOFS";
			this->delta = 0;

			this->abs_weightVec.resize(this->weightDim);
		}

		virtual ~FOFS(){
		}

		/**
		 * @Synopsis inherited functions
		 */
	public:
		/**
		 * PrintModelSettings print the info of optimization algorithm
		 */
		virtual void PrintModelSettings() const {
			OnlineFeatureSelection<FeatType, LabelType>::PrintModelSettings();
			printf("\tdelta:\t%.2f\n", this->delta);
		}

		/**
		 * @Synopsis SetParameter set parameters for the learning model
		 *
		 * @Param param
		 */
		virtual void SetParameter(BOC::Params &param){
			OnlineFeatureSelection<FeatType, LabelType>::SetParameter(param);
			this->delta = param.FloatValue("-delta");
			INVALID_ARGUMENT_EXCEPTION(delta, this->delta >= 0, "no smaller than 0");
		}

		/**
		 * @Synopsis BeginTrain Reset the optimizer to the initialization status of training
		 */
		virtual void BeginTrain() {
			OnlineFeatureSelection<FeatType, LabelType>::BeginTrain();

			this->w_norm = 0;
			this->norm_coeff = 1.f / sqrtf(this->delta);

			this->shrinkage = 1.f - this->delta * this->eta0;
			this->power_t = 0;

			if (this->K > 0){
				if (this->weightDim < this->K + 1)
					this->UpdateModelDimention(this->K);
				this->minHeap.Init(this->weightDim - 1, this->K, this->abs_weightVec.begin + 1);
			}
		}

		/**
		 * @Synopsis Iterate Iteration of online learning
		 *
		 * @Param x current input data example
		 *
		 * @Returns  prediction of the current example
		 */
		virtual float Iterate(const DataPoint<FeatType, LabelType> &x) {
			//we use the oposite of w
			float y = this->TrainPredict(this->weightVec, x);
			size_t featDim = x.indexes.size();

			float gt_i = this->lossFunc->GetGradient(x.label, y);
			if (gt_i == 0){
				return y;
			}

			//update with sgd
			for (size_t i = 0; i < featDim; i++) {
				this->weightVec[x.indexes[i]] -= this->eta0 * gt_i * x.features[i];
			}
			//update bias 
			this->weightVec[0] -= this->eta0 * gt_i;

			w_norm = 0;
			for (size_t i = 0; i < this->weightDim; i++)
				w_norm += this->weightVec[i] * this->weightVec[i];
			//shrinkage
			float coeff = this->norm_coeff / sqrtf(w_norm);
			if (coeff < 1){
				for (IndexType i = 0; i < this->weightDim; i++){
					this->weightVec[i] *= coeff;
				}
			}
			if (this->K > 0){
				//truncate
				for (IndexType i = 0; i < this->weightDim; i++)
					this->abs_weightVec[i] = fabs(this->weightVec[i]);

				this->minHeap.BuildHeap();
				//truncate
				IndexType ret_id;
				for (IndexType i = 0; i < this->weightDim - 1; i++){
					if (this->minHeap.UpdateHeap(i, ret_id) == true){
						this->weightVec[ret_id + 1] = 0;
						this->abs_weightVec[ret_id + 1] = 0;
					}
				}
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
				this->abs_weightVec.reserve(new_dim + 1);
				this->abs_weightVec.resize(new_dim + 1);
				this->abs_weightVec.zeros(this->abs_weightVec.begin + this->weightDim, this->abs_weightVec.end);

				this->minHeap.UpdateDataNum(new_dim, this->abs_weightVec.begin + 1);

				OnlineFeatureSelection<FeatType, LabelType>::UpdateModelDimention(new_dim);
			}
		}
	};

	IMPLEMENT_MODEL_CLASS(FOFS, "First Order Online Feature Selection")
}

#endif
