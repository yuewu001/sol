/*************************************************************************
  > File Name: PET.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Tue 17 Dec 2013 04:59:06 AM EST
  > Descriptions: Online Feature Selection with PE_trunc
  > Reference: Online Feature Selection and its applications
  ************************************************************************/
#ifndef HEADER_OPTIMISER_PET
#define HEADER_OPTIMISER_PET

#include "SparseOnlineLinearModel.h"
#include "../../../../utils/MinHeap.h"

/**
*  namespace: Batch and Online Classification
*/
namespace BOC {
	template <typename FeatType, typename LabelType>
	class PET : public SparseOnlineLinearModel<FeatType, LabelType> {

		DECLARE_CLASS

	protected:
		IndexType K; //keep top K elemetns

		s_array<float> abs_weightVec;

		MinHeap<float> minHeap;

		float(*pEta_time)(size_t t, float pt);
	public:
		PET(LossFunction<FeatType, LabelType> *lossFunc) :
			SparseOnlineLinearModel<FeatType, LabelType>(lossFunc) {
				this->K = 0;

				this->abs_weightVec.resize(this->weightDim);
			}

		virtual ~PET() {
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
			printf("\tK:\t%d\n", this->K);
		}

		/**
		 * @Synopsis SetParameter set parameters for the learning model
		 *
		 * @Param param
		 */
		virtual void SetParameter(BOC::Params &param){
			OnlineLinearModel<FeatType, LabelType>::SetParameter(param);
			this->K = param.IntValue("-k");
		}

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
			this->curIterNum++;
			float y = this->Predict(x);
			size_t featDim = x.indexes.size();
			this->eta = this->eta0 / this->pEta_time(this->curIterNum, this->power_t);

			float gt_i = this->lossFunc->GetGradient(x.label, y);
			if (gt_i == 0){
				return y;
			}

			IndexType index_i = 0;
			//update with sgd
			for (size_t i = 0; i < featDim; i++) {
				index_i = x.indexes[i];
				this->weightVec[index_i] -= this->eta * gt_i * x.features[i];
				this->abs_weightVec[index_i] = fabsf(this->weightVec[index_i]);
			}
			//update bias 
			this->weightVec[0] -= this->eta * gt_i;
			if (this->K > 0){
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

				SparseOnlineLinearModel<FeatType, LabelType>::UpdateModelDimention(new_dim);
			}
		}
	};

	IMPLEMENT_MODEL_CLASS(PET,"Perceptron with Truncation")
}

#endif
