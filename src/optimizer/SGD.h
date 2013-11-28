/*************************************************************************
  > File Name: SGD.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: 8/19/2013 Monday 10:37:08 AM
  > Functions: Stochastic Gradient Descent
 ************************************************************************/

#pragma once

#include "../common/global.h"
#include "Optimizer.h"

namespace SOL {
	template <typename FeatType, typename LabelType>
	class SGD: public Optimizer<FeatType, LabelType> {
	protected:
		float (*pEta_time)(size_t t, float pt);
	public:
		SGD(DataSet<FeatType, LabelType> &dataset, LossFunction<FeatType, LabelType> &lossFunc,
			NormType type = NormType_None);
		virtual ~SGD();

	protected:
		//Reset the optimizer to the initialization status of training
		virtual void BeginTrain();
	protected:
		//this is the core of different updating algorithms
		virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
	};

	template <typename FeatType, typename LabelType>
	SGD<FeatType, LabelType>::SGD(DataSet<FeatType, LabelType> &dataset, 
		LossFunction<FeatType, LabelType> &lossFunc,
		NormType type): Optimizer<FeatType, LabelType>(dataset, lossFunc) {
			this->id_str = "SGD";
	}

	template <typename FeatType, typename LabelType>
	SGD<FeatType, LabelType>::~SGD() {
	}

	//update weight vector with stochastic gradient descent
	template <typename FeatType, typename LabelType>
	float SGD<FeatType,LabelType>::UpdateWeightVec(
		const DataPoint<FeatType, LabelType> &x) {
			this->eta = this->eta0 / this->pEta_time(this->curIterNum, this->power_t);

			float y = this->Predict(x);
			size_t featDim = x.indexes.size();
			float gt_i = this->lossFunc->GetGradient(x.label,y);

			for (size_t i = 0; i < featDim; i++)
				this->weightVec[x.indexes[i]] -= this->eta * gt_i * x.features[i];

			//update bias 
			this->weightVec[0] -= this->eta * gt_i;
			return y;
	}
	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void SGD<FeatType, LabelType>::BeginTrain() {
		Optimizer<FeatType, LabelType>::BeginTrain();

		if (this->power_t == 0.5)
			this->pEta_time = pEta_sqrt;
		else if(this->power_t == 0)
			this->pEta_time = pEta_const;
		else if (this->power_t == 1)
			this->pEta_time = pEta_linear;
		else
			this->pEta_time = pEta_general;
	}
}
