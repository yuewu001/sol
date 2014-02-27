/*************************************************************************
  > File Name: PET.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Tue 17 Dec 2013 04:59:06 AM EST
  > Descriptions: Online Feature Selection with PE_trunc
  > Reference: Online Feature Selection and its applications
 ************************************************************************/
#ifndef HEADER_OPTIMISER_PET
#define HEADER_OPTIMISER_PET

#include "../utils/util.h"
#include "Optimizer.h"
#include "MinHeap.h"
#include <algorithm>
#include <math.h>
#include <queue>

namespace SOL {
	template <typename FeatType, typename LabelType>
	class PET: public Optimizer<FeatType, LabelType> {
	protected:
		IndexType K; //keep top K elemetns

		s_array<float> abs_weightVec;

		MinHeap<float> minHeap;

		float (*pEta_time)(size_t t, float pt);
	public:
		PET(DataSet<FeatType, LabelType> &dataset, 
			LossFunction<FeatType, LabelType> &lossFunc);
		virtual ~PET();

	public:
		void SetParameterEx(int k);
		/**
		* PrintOptInfo print the info of optimization algorithm
		*/
		virtual void PrintOptInfo() const {
			Optimizer<FeatType,LabelType>::PrintOptInfo();
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
	};

	template <typename FeatType, typename LabelType>
	PET<FeatType, LabelType>::PET(DataSet<FeatType, LabelType> &dataset, 
		LossFunction<FeatType, LabelType> &lossFunc):
	Optimizer<FeatType, LabelType>(dataset, lossFunc){
		this->id_str = "PET";
		this->K = 0;

		this->abs_weightVec.resize(this->weightDim);
	}

	template <typename FeatType, typename LabelType>
	PET<FeatType, LabelType>::~PET() {
	}

	//this is the core of different updating algorithms
	//return the predict
	template <typename FeatType, typename LabelType>
	float PET<FeatType,LabelType>::UpdateWeightVec(
		const DataPoint<FeatType, LabelType> &x) {
			//we use the oposite of w
			float y = this->Predict(x);
			size_t featDim = x.indexes.size();
			this->eta = this->eta0 / this->pEta_time(this->curIterNum, this->power_t);

			float gt_i = this->lossFunc->GetGradient(x.label,y);
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
				/*
				//test if top K is in
				float min_top = 100000;
				float max_not_top = -10000;
				for (IndexType i = 1; i < this->weightDim ; i++){
					if (this->minHeap.is_topK(i - 1) == true){
						min_top = min_top > this->abs_weightVec[i] ? this->abs_weightVec[i] : min_top;
					}
					else{
						max_not_top = max_not_top < this->abs_weightVec[i] ? this->abs_weightVec[i] : max_not_top;
						this->weightVec[i] = 0;
						this->abs_weightVec[i] = 0;
					}
				}
				*/
			}
			return y;
	}

	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void PET<FeatType, LabelType>::BeginTrain() {
		Optimizer<FeatType, LabelType>::BeginTrain();
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
				this->UpdateWeightSize(this->K);
			this->minHeap.Init(this->weightDim - 1, this->K, this->abs_weightVec.begin + 1);
		}
	}

	//called when a train ends
	template <typename FeatType, typename LabelType>
	void PET<FeatType, LabelType>::EndTrain() {
		Optimizer<FeatType, LabelType>::EndTrain();
	}

	template <typename FeatType, typename LabelType>
	void PET<FeatType, LabelType>::SetParameterEx(int k) {
		this->K = k > 0 ? k : this->K;
	}

	//Change the dimension of weights
	template <typename FeatType, typename LabelType>
	void PET<FeatType, LabelType>::UpdateWeightSize(IndexType newDim) {
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
