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
	public:
		SGD(DataSet<FeatType, LabelType> &dataset, LossFunction<FeatType, LabelType> &lossFunc,
			NormType type = NormType_None);
		virtual ~SGD();

	protected:
		//this is the core of different updating algorithms
		virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
	};

	template <typename FeatType, typename LabelType>
	SGD<FeatType, LabelType>::SGD(DataSet<FeatType, LabelType> &dataset, 
		LossFunction<FeatType, LabelType> &lossFunc,
			NormType type): Optimizer<FeatType, LabelType>(dataset, lossFunc) {
        this->id_str = "SGD";
        this->sparse_soft_thresh = init_sparse_soft_thresh;
	}

	template <typename FeatType, typename LabelType>
	SGD<FeatType, LabelType>::~SGD() {
	}

	//update weight vector with stochastic gradient descent
	template <typename FeatType, typename LabelType>
	float SGD<FeatType,LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x) {
		float y = this->Predict(x);
        int featDim = x.indexes.size();
        float gt_i = this->lossFunc->GetGradient(x.label,y);

        for (int i = 0; i < featDim; i++)
            this->weightVec[x.indexes[i]] -= this->eta * gt_i * x.features[i];

		//update bias 
		this->weightVec[0] -= this->eta * gt_i;
		return y;
	}
}
