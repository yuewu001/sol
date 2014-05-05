/*************************************************************************
  > File Name: FOFS.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Tue 17 Dec 2013 04:59:06 AM EST
  > Descriptions: Online Feature Selection
  > Reference: Online Feature Selection and its applications
 ************************************************************************/
#ifndef HEADER_OPTIMISER_FOFS
#define HEADER_OPTIMISER_FOFS

#include "../utils/util.h"
#include "MinHeap.h"

#include "Optimizer.h"
#include <algorithm>
#include <math.h>
#include <queue>

namespace SOL {
	template <typename FeatType, typename LabelType>
	class FOFS: public Optimizer<FeatType, LabelType> {
	protected:
		IndexType K; //keep top K elemetns

		float w_norm;
		float norm_coeff;

		float shrinkage;
		float delta;
		s_array<float> abs_weightVec;

		MinHeap<float> minHeap;

	public:
		FOFS(DataSet<FeatType, LabelType> &dataset, 
			LossFunction<FeatType, LabelType> &lossFunc);
		virtual ~FOFS();

	public:
		void SetParameterEx(int k, float delta);
		/**
		* PrintOptInfo print the info of optimization algorithm
		*/
		virtual void PrintOptInfo() const {
			printf("--------------------------------------------------\n");
			printf("Algorithm: %s\n\n",this->Id_Str().c_str());
			printf("eta:\t%.2f\n", this->eta0);
			printf("delta:\t%.2f\n", this->delta);
			printf("K:\t%d\n\n", this->K);
		}
	protected:
		//this is the core of different updating algorithms
		virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
		//reset the optimizer to this initialization
		virtual void BeginTrain();
		//called when a train ends
		virtual void EndTrain();

		//Change the dimension of weights
		virtual void UpdateWeightSize(IndexType newDim);

	protected:
		void QuickSort(float *a, IndexType low, IndexType high, IndexType *m_index);
		void Sort(float *a, IndexType low, IndexType high, IndexType *m_index);
	};

	template <typename FeatType, typename LabelType>
	FOFS<FeatType, LabelType>::FOFS(DataSet<FeatType, LabelType> &dataset, 
		LossFunction<FeatType, LabelType> &lossFunc):
	Optimizer<FeatType, LabelType>(dataset, lossFunc){
		this->id_str = "First Order Online Feature Selection";
		this->K = 0;
		this->delta = init_ofs_delta;
		this->eta0 = init_ofs_eta;

		this->abs_weightVec.resize(this->weightDim);
	}

	template <typename FeatType, typename LabelType>
	FOFS<FeatType, LabelType>::~FOFS() {
	}

	//this is the core of different updating algorithms
	//return the predict
	template <typename FeatType, typename LabelType>
	float FOFS<FeatType,LabelType>::UpdateWeightVec(
		const DataPoint<FeatType, LabelType> &x) {
			//we use the oposite of w
			float y = this->Predict(x);
			size_t featDim = x.indexes.size();
			//this->eta = this->eta0;

			/*
			//shrinkage
			for (size_t i = 0; i < this->weightDim; i++){
				this->weightVec[i] *= shrinkage;
			}
			*/

			float gt_i = this->lossFunc->GetGradient(x.label,y);
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

	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void FOFS<FeatType, LabelType>::BeginTrain() {
		Optimizer<FeatType, LabelType>::BeginTrain();
		this->w_norm = 0;
		this->norm_coeff = 1.f / sqrtf(this->delta);

		this->shrinkage = 1.f - this->delta * this->eta0;
		this->power_t = 0;
		
		if (this->K > 0){
			if (this->weightDim < this->K + 1)
				this->UpdateWeightSize(this->K);
			this->minHeap.Init(this->weightDim - 1, this->K, this->abs_weightVec.begin + 1);
		}
	}

	//called when a train ends
	template <typename FeatType, typename LabelType>
	void FOFS<FeatType, LabelType>::EndTrain() {
		Optimizer<FeatType, LabelType>::EndTrain();
	}

	template <typename FeatType, typename LabelType>
	void FOFS<FeatType, LabelType>::SetParameterEx(int k, float Delta) {
		this->K = k > 0 ? k : this->K;
		this->delta = Delta > 0 ? Delta : this->delta;
	}

	//Change the dimension of weights
	template <typename FeatType, typename LabelType>
	void FOFS<FeatType, LabelType>::UpdateWeightSize(IndexType newDim) {
		if (newDim < this->weightDim)
			return;
		else {
			this->abs_weightVec.reserve(newDim + 1);
			this->abs_weightVec.resize(newDim + 1);
			this->abs_weightVec.zeros(this->abs_weightVec.begin + this->weightDim, this->abs_weightVec.end);

			this->minHeap.UpdateDataNum(newDim, this->abs_weightVec.begin + 1);

			Optimizer<FeatType, LabelType>::UpdateWeightSize(newDim);
		}
	}
}

#endif
