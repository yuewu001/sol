/*************************************************************************
  > File Name: PET.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Tue 17 Dec 2013 04:59:06 AM EST
  > Descriptions: Online Feature Selection with PE_trunc
  > Reference: Online Feature Selection and its applications
  ************************************************************************/
#ifndef HEADER_OPTIMISER_PET
#define HEADER_OPTIMISER_PET

#include "OnlineFeatureSelection.h"
#include "../../../../utils/MinHeap.h"

/**
*  namespace: Batch and Online Classification
*/
namespace BOC {
	template <typename FeatType, typename LabelType>
	class PET : public OnlineFeatureSelection < FeatType, LabelType > {

		DECLARE_CLASS

#pragma region Class Members
	protected:
		s_array<float> weightMatrixPNorm; //p-norm of weight matrix

		MinHeap<float> minHeap;

		float(*pEta_time)(size_t t, float pt);
#pragma endregion Class Members

#pragma region Constructors and Basic Functions
	public:
		PET(LossFunction<FeatType, LabelType> *lossFunc, int classNum) :
			OnlineFeatureSelection<FeatType, LabelType>(lossFunc, classNum) {
			this->modelName = "PET";
			this->weightMatrixPNorm.resize(this->weightDim);
		}

		virtual ~PET() {
		}
#pragma endregion Constructors and Basic Functions

#pragma region Train Related
	public:
		/**
		 * @Synopsis BeginTrain Reset the optimizer to the initialization status of training
		 */
		virtual void BeginTrain() {
			OnlineFeatureSelection<FeatType, LabelType>::BeginTrain();

			if (this->power_t == 0.5)
				this->pEta_time = pEta_sqrt;
			else if (this->power_t == 0)
				this->pEta_time = pEta_const;
			else if (this->power_t == 1)
				this->pEta_time = pEta_linear;
			else
				this->pEta_time = pEta_general;

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
				this->weightMatrixPNorm.reserve(new_dim + 1);
				this->weightMatrixPNorm.resize(new_dim + 1);
				this->weightMatrixPNorm.zeros(this->weightMatrixPNorm.begin + this->weightDim, this->weightMatrixPNorm.end);

				this->minHeap.UpdateDataNum(new_dim, this->weightMatrixPNorm.begin + 1);

				OnlineFeatureSelection<FeatType, LabelType>::UpdateModelDimention(new_dim);
			}
		}

		/**
		 * @Synopsis IterateBC Iteration of online learning for binary classification
		 *
		 * @Param x current input data example
		 *
		 * @Returns  predicted class of the current example
		 */
		virtual int IterateBC(const DataPoint<FeatType, LabelType> &x, float* predict){
			this->curIterNum++;
			*predict = this->TrainPredict(*this->pWeightVecBC, x);
			int label = this->GetClassLabel(x);
			float gt = 0;
			this->lossFunc->GetGradient(label, predict, &gt);

			if (gt != 0){
				this->UpdateWeightVec(x, *this->pWeightVecBC, gt);

				//update pnorm
				size_t featDim = x.indexes.size();
				for (size_t i = 0; i < featDim; ++i){
					this->weightMatrixPNorm[x.indexes[i]] = (*this->pWeightVecBC)[x.indexes[i]] * (*this->pWeightVecBC)[x.indexes[i]];
				}

				if (this->K > 0){
					this->minHeap.BuildHeap();
					//truncate
					IndexType ret_id;
					for (IndexType i = 0; i < this->weightDim - 1; i++){
						if (this->minHeap.UpdateHeap(i, ret_id) == true){
							(*this->pWeightVecBC)[ret_id + 1] = 0;
							this->weightMatrixPNorm[ret_id + 1] = 0;
						}
					}
				}
			}

			if (this->IsCorrect(label, predict) == false){
				return -label;
			}
			else{
				return x.label;
			}
		}

		/**
		 * @Synopsis IterateMC Iteration of online learning for multiclass classification
		 *
		 * @Param x current input data example
		 *
		 * @Returns  predicted class of the current example
		 */
		virtual int IterateMC(const DataPoint<FeatType, LabelType> &x, float* predict){
			this->curIterNum++;
			for (int k = 0; k < this->classfier_num; ++k){
				this->mc_predicts[k] = this->TrainPredict(this->weightMatrix[k], x);
			}

			this->mc_gradients;

			//not correct
			if (false){
				for (int k = 0; k < this->classfier_num; ++k){
					this->UpdateWeightVec(x, this->weightMatrix[k], this->mc_gradients[k]);
				}

				//update pnorm
				size_t featDim = x.indexes.size();
				for (size_t i = 0; i < featDim; ++i){
					this->weightMatrixPNorm[x.indexes[i]] = 0;
				}

				for (int k = 0; k < this->classfier_num; ++k){
					s_array<float> weightVec = this->weightMatrix[k];
					for (size_t i = 0; i < featDim; ++i){
						this->weightMatrixPNorm[x.indexes[i]] += weightVec[x.indexes[i]] * weightVec[x.indexes[i]];
					}
				}

				if (this->K > 0){
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
				return (std::max_element(this->mc_predicts.begin(), this->mc_predicts.end()) - this->mc_predicts.begin());
			}
			return x.label;
		}

	protected:
		/**
		 * @Synopsis UpdateWeightVec Update the weight vector
		 *
		 * @Param x current input data example
		 * @Param weightVec weight vector to be updated
		 * @param gt common part of the gradient
		 * @Param beta extra multiplier for updating, if none, set it to 1
		 *
		 */
		virtual void UpdateWeightVec(const DataPoint<FeatType, LabelType> &x, s_array<float>& weightVec, float gt){
			this->eta = this->eta0 / this->pEta_time(this->curIterNum, this->power_t);
			size_t featDim = x.indexes.size();

			//update with sgd
			for (size_t i = 0; i < featDim; i++) {
				weightVec[x.indexes[i]] -= this->eta * gt * x.features[i];
			}

			//update bias 
			weightVec[0] -= this->eta * gt;
		}

	};

	IMPLEMENT_MODEL_CLASS(PET, "Perceptron with Truncation")
}

#endif
