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

		float delta;
		s_array<float> weightMatrixPNorm;

		MinHeap<float> minHeap;

	public:
		FOFS(LossFunction<FeatType, LabelType> *lossFunc, int classNum) :
			OnlineFeatureSelection<FeatType, LabelType>(lossFunc, classNum) {
			this->modelName = "FOFS";
			this->delta = 0;
			this->power_t = 0;

			this->weightMatrixPNorm.resize(this->weightDim);
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
			float val = param.FloatValue("-delta");
			if (val >= 0) {
				this->delta = val;
			}
			this->power_t = 0;
		}

		/**
		 * @Synopsis BeginTrain Reset the optimizer to the initialization status of training
		 */
		virtual void BeginTrain() {
			INVALID_ARGUMENT_EXCEPTION(delta, this->delta >= 0, "no smaller than 0");
			OnlineFeatureSelection<FeatType, LabelType>::BeginTrain();

			this->w_norm = 0;
			this->norm_coeff = 1.f / sqrtf(this->delta);

			this->power_t = 0;

			this->weightMatrixPNorm.zeros();
			if (this->K > 0){
				if (this->weightDim < this->K + 1)
					this->UpdateModelDimention(this->K);
				this->minHeap.Init(this->weightDim - 1, this->K, this->weightMatrixPNorm.begin + 1);
			}
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
				this->weightMatrixPNorm.resize(new_dim + 1);
				this->weightMatrixPNorm.zeros(this->weightMatrixPNorm.begin + this->weightDim, this->weightMatrixPNorm.end);

				this->minHeap.UpdateDataNum(new_dim, this->weightMatrixPNorm.begin + 1);

				OnlineFeatureSelection<FeatType, LabelType>::UpdateModelDimention(new_dim);
			}
		}

	protected:
		/**
		 * @Synopsis UpdateWeightVec Update the weight vector
		 *
		 * @Param x current input data example
		 * @Param gt common part of the gradient
		 *
		 */
		virtual void UpdateWeightVec(const DataPoint<FeatType, LabelType> &x, float* gt_t){
			size_t featDim = x.indexes.size();

			//update with sgd
			for (int k = 0; k < this->classfier_num; ++k){
				s_array<float> &weightVec = this->weightMatrix[k];
				for (size_t i = 0; i < featDim; i++) {
					weightVec[x.indexes[i]] -= this->eta0 * gt_t[k] * x.features[i];
				}
				//update bias 
				weightVec[0] -= this->eta0 * gt_t[k];

				float w_norm = 0;
				for (IndexType i = 0; i < this->weightDim; i++)
					w_norm += weightVec[i] * weightVec[i];

				float coeff = this->norm_coeff / sqrtf(w_norm);
				if (coeff < 1){
					for (IndexType i = 0; i < this->weightDim; i++){
						weightVec[i] *= coeff;
					}
				}
			}

			if (this->K > 0){

				//update pnorm
				for (size_t i = 0; i < this->weightDim; ++i){
					this->weightMatrixPNorm[i] = 0;
				}

				for (int k = 0; k < this->classfier_num; ++k){
					s_array<float> &weightVec = this->weightMatrix[k];
					for (size_t i = 0; i < this->weightDim; ++i){
						this->weightMatrixPNorm[i] += weightVec[i] * weightVec[i];
					}
				}

				this->minHeap.BuildHeap();
				//truncate
				IndexType ret_id;
				for (IndexType i = 0; i < this->weightDim - 1; i++){
					if (this->minHeap.UpdateHeap(i, ret_id) == true){
						for (int k = 0; k < this->classfier_num; ++k){
							(this->weightMatrix[k])[ret_id + 1] = 0;
						}
						this->weightMatrixPNorm[ret_id + 1] = 0;
					}
				}
			}
		}
	};

	IMPLEMENT_MODEL_CLASS(FOFS, "First Order Online Feature Selection")
}

#endif
