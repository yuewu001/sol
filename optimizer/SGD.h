/*************************************************************************
> File Name: SGD.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 8/19/2013 Monday 10:37:08 AM
> Functions: Stochastic Gradient Descent
************************************************************************/

#pragma once

#include "../common/global.h"
#include "Optimizer.h"

namespace SOL
{
	template <typename FeatType, typename LabelType>
	class SGD: public Optimizer<FeatType, LabelType>
	{
	public:
		SGD(DataSet<FeatType, LabelType> &dataset, LossFunction<FeatType, LabelType> &lossFunc,
			NormType type = NormType_None);
		virtual ~SGD();

	protected:
		//this is the core of different updating algorithms
		virtual double UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
    protected:
		NormType normType;
	};

	template <typename FeatType, typename LabelType>
	SGD<FeatType, LabelType>::SGD(DataSet<FeatType, LabelType> &dataset, 
		LossFunction<FeatType, LabelType> &lossFunc,
			NormType type): Optimizer<FeatType, LabelType>(dataset, lossFunc) 
	{
		this->normType = type;

        //sparse soft threshold
        this->sparse_soft_thresh = 1e-6;
	}

	template <typename FeatType, typename LabelType>
	SGD<FeatType, LabelType>::~SGD()
	{
	}

	//update weight vector with stochastic gradient descent
	template <typename FeatType, typename LabelType>
	double SGD<FeatType,LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
	{
		double y = this->Predict(x);
        int featDim = x.indexes.size();
        double gt_i = this->lossFunc->GetGradient(x,y);

        if(this->normType == NormType_L1) //normalize with L1
        {
            double coeff = this->eta * this->lambda;
            for(int i = 1; i < this->weightDim; i++)
                this->weightVec[i] -= coeff * Sgn(this->weightVec[i]);
        }

        for (int i = 0; i < featDim; i++)
            this->weightVec[x.indexes[i]] -= this->eta * gt_i * x.features[i];

		//update bias 
		this->weightVec[0] -= this->eta * gt_i;
		return y;
	}
}
