/*************************************************************************
> File Name: SGD.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 8/19/2013 Monday 10:37:08 AM
> Functions: Stochastic Gradient Descent
************************************************************************/

#pragma once

#include "../global.h"
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
		//called when a pass ended
		virtual void PassEnd() {} 
		//called when a round ended
		virtual void RoundEnd() {}
		//reset the optimizer to this initialization
		virtual void Reset();


	public:
		//Change the dimension of weights
		virtual void UpdateWeightSize(int newDim);
	protected:
		//for fast calculation
		size_t*timeStamp;
	protected:
		NormType normType;
	};

	template <typename FeatType, typename LabelType>
	SGD<FeatType, LabelType>::SGD(DataSet<FeatType, LabelType> &dataset, 
		LossFunction<FeatType, LabelType> &lossFunc,
			NormType type): Optimizer<FeatType, LabelType>(dataset, lossFunc) 
	{
		this->normType = type;

		switch(this->normType)
		{
		case NormType_L1:
			this->timeStamp = new size_t[this->weightDim];
			break;
		default:
			this->timeStamp = NULL;
			break;
		}
		this->Reset();
	}

	template <typename FeatType, typename LabelType>
	SGD<FeatType, LabelType>::~SGD()
	{
		if (this->timeStamp != NULL)
			delete []this->timeStamp;
	}

	//update weight vector with stochastic gradient descent
	template <typename FeatType, typename LabelType>
	double SGD<FeatType,LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
	{
		int featDim = x.Dim();

		switch(this->normType)
		{
		case NormType_L1: //normalize with L1
			for (int i = 0; i < featDim; i++)
				this->weightVec[i] -= this->eta * this->lambda * Sgn(this->weightVec[i]);
			break;
		default:
			break;
		}
		double y = this->Predict(x);

		for (int i = 0; i < featDim; i++)
		{
			if (x[i] != 0)
			{
				double gt_i = this->lossFunc->GetGradient(x,y,i);
				this->weightVec[i] -= this->eta * gt_i;
			}
		}

		//update bias 
		double gt_i = this->lossFunc->GetBiasGradient(x,y);
		this->weightVec[this->weightDim - 1] -= this->eta * gt_i;
		return y;
	}
	
	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void SGD<FeatType, LabelType>::Reset()
	{
		Optimizer<FeatType, LabelType>::Reset();

		if (this->timeStamp != NULL) //reset time stamp
			memset(this->timeStamp,0,sizeof(size_t) * this->weightDim);
	}

	//Change the dimension of weights
	template <typename FeatType, typename LabelType>
	void SGD<FeatType, LabelType>::UpdateWeightSize(int newDim)
	{
		if (newDim < this->weightDim - 1)
			return;
		else
		{
			if (this->normType == NormType_L1)
			{
				size_t* newT = new size_t[newDim + 1];
				memset(newT,0,sizeof(size_t) * (newDim + 1));
				memcpy(newT,this->timeStamp,sizeof(size_t) * (this->weightDim - 1));
				newT[newDim] = this->timeStamp[this->weightDim - 1];
				delete []this->timeStamp;
				this->timeStamp = newT;
			}
			Optimizer<FeatType,LabelType>::UpdateWeightSize(newDim);
		}
	}
}
