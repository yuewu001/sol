/*************************************************************************
> File Name: STG.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/18 星期日 17:25:54
> Functions: Sparse Online Learning With Truncated Gradient
> Reference:
Langford J, Li L, Zhang T. Sparse online learning via truncated 
gradient[J]. The Journal of Machine Learning Research, 2009, 10: 
777-801. 
************************************************************************/

#pragma once


#include "../utils/util.h"
#include "Optimizer.h"
#include <math.h>
#include <limits>

namespace SOL {
	template <typename FeatType, typename LabelType>
	class STG: public Optimizer<FeatType, LabelType> {
	protected:
		int K;

	protected:
		s_array<size_t> timeStamp;
	protected:
		float (*pEta_time)(size_t t, float pt);

	public:
		STG(DataSet<FeatType, LabelType> &dataset, 
			LossFunction<FeatType, LabelType> &lossFunc);
		virtual ~STG();

	public:
		void SetParameterEx(int K = -1);
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
	STG<FeatType, LabelType>::STG(DataSet<FeatType, LabelType> &dataset, 
		LossFunction<FeatType, LabelType> &lossFunc):
	Optimizer<FeatType, LabelType>(dataset, lossFunc){
		this->id_str = "STG";
		this->K = init_k;
		this->timeStamp.resize(this->weightDim);
	}

	template <typename FeatType, typename LabelType>
	STG<FeatType, LabelType>::~STG() {
	}

	//this is the core of different updating algorithms
	//return the predict
	template <typename FeatType, typename LabelType>
	float STG<FeatType,LabelType>::UpdateWeightVec(
		const DataPoint<FeatType, LabelType> &x) {
			this->eta = this->eta0 / this->pEta_time(this->curIterNum, this->power_t);

			size_t featDim = x.indexes.size();
			float alpha = this->eta * this->lambda;

			float y = this->Predict(x); 
			float gt_i = this->lossFunc->GetGradient(x.label,y) * this->eta;

			size_t stepK = 0;
			for (size_t i = 0; i < featDim; i++) {
				IndexType index_i = x.indexes[i];
				//update the weight
				this->weightVec[index_i] -= gt_i * x.features[i];

				//lazy update the weight
				//truncated gradient
				if (this->timeStamp[index_i] == 0) {
					this->timeStamp[index_i] = this->curIterNum;
					continue;
				}
				else{
					stepK = this->curIterNum - this->timeStamp[index_i];
					if (stepK < size_t(this->K))
						continue;

					stepK -= stepK % this->K;
					this->timeStamp[index_i] += stepK;
					this->weightVec[index_i] = trunc_weight(this->weightVec[index_i],
						stepK * alpha);
				}
			}
			//bias term
			this->weightVec[0] -= gt_i;
			return y;
	}


	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void STG<FeatType, LabelType>::BeginTrain() {
		Optimizer<FeatType, LabelType>::BeginTrain();
		//reset time stamp
		this->timeStamp.zeros();

		if (this->power_t == 0.5)
			this->pEta_time = pEta_sqrt;
		else if(this->power_t == 0)
			this->pEta_time = pEta_const;
		else if (this->power_t == 1)
			this->pEta_time = pEta_linear;
		else
			this->pEta_time = pEta_general;
	}

	//called when a train ends
	template <typename FeatType, typename LabelType>
	void STG<FeatType, LabelType>::EndTrain() {
		for (IndexType index_i = 1; index_i < this->weightDim; index_i++) {
			//truncated gradient
			size_t stepK = this->curIterNum - this->timeStamp[index_i];
			stepK -= stepK % this->K;

			this->weightVec[index_i] = trunc_weight(this->weightVec[index_i],
				stepK * this->lambda * this->eta);
		}
		Optimizer<FeatType, LabelType>::EndTrain();
	}

	template <typename FeatType, typename LabelType>
	void STG<FeatType, LabelType>::SetParameterEx(int k) {
		this->K = k > 0 ? k : this->K;
	}

	//Change the dimension of weights
	template <typename FeatType, typename LabelType>
	void STG<FeatType, LabelType>::UpdateWeightSize(IndexType newDim) {
		if (newDim < this->weightDim)
			return;
		else {
			this->timeStamp.reserve(newDim + 1);
			this->timeStamp.resize(newDim + 1);
			//set the rest to zero
			this->timeStamp.zeros(this->timeStamp.begin + this->weightDim,
				this->timeStamp.end);

			Optimizer<FeatType,LabelType>::UpdateWeightSize(newDim);
		}
	}
}
