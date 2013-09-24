/*************************************************************************
> File Name: FOBOS.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/20 Tuesday 11:14:54
> Functions: FOBOS: Efficient Online Batch Learning Using 
					Forward Backward Splitting

> Reference:
		Duchi J, Singer Y. Efficient online and batch learning using 
		forward backward splitting[J]. The Journal of Machine Learning 
		Research, 2009, 10: 2899-2934.

		This file implements the L1 and L2 square regularization
 ************************************************************************/

#pragma once
#include "../global.h"
#include "../util.h"

#include "Optimizer.h"


#include <cmath>
#include <limits>

namespace SOL
{
	template <typename FeatType, typename LabelType>
	class FOBOS: public Optimizer<FeatType, LabelType>
	{
	public:
		FOBOS(DataSet<FeatType, LabelType> &dataset, 
			LossFunction<FeatType, LabelType> &lossFunc, NormType normType = NormType_L1);
		~FOBOS();

	public:
		//Change the dimension of weights
		virtual void UpdateWeightSize(int newDim);

	protected:
		//this is the core of different updating algorithms
		virtual double UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
		double UpdateWeightVec_L1(const DataPoint<FeatType, LabelType> &x);
		double UpdateWeightVec_L2S(const DataPoint<FeatType, LabelType> &x);

		//reset
		virtual void Reset();
		//called when a pass ends
		virtual void PassEnd();
		//called when a round ended
		virtual void RoundEnd(); 

	protected:
		NormType normType;

	protected:
		size_t *timeStamp;
	};

	template <typename FeatType, typename LabelType>
	FOBOS<FeatType, LabelType>::FOBOS(DataSet<FeatType, LabelType> &dataset, 
		LossFunction<FeatType, LabelType> &lossFunc, NormType normType):
	Optimizer<FeatType, LabelType>(dataset, lossFunc), timeStamp(NULL)
	{
		this->normType = normType;
		this->timeStamp = new size_t[this->weightDim];
		this->Reset();
	}

	template <typename FeatType, typename LabelType>
	FOBOS<FeatType, LabelType>::~FOBOS()
	{
		if(this->timeStamp != NULL)
			delete []this->timeStamp;
	}

	//this is the core of different updating algorithms
	template <typename FeatType, typename LabelType>
	double FOBOS<FeatType,LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
	{
		switch(normType)
		{
			case NormType_L1:
				return this->UpdateWeightVec_L1(x);
				break;
			case NormType_L2S:
				return this->UpdateWeightVec_L2S(x);
				break;
			default:
				throw invalid_argument("Unsupported update rule!");
				break;
		}
		
	}

	/**
	 *  UpdateWeightVec_L1: use L1 norm as the regularization term
	 *  r(w) = lambda * |w|
	 *
	 * @tparam FeatType
	 * @tparam LabelType
	 * @Param:  x
	 * @Param:  y
	 */
	template <typename FeatType, typename LabelType>
	double FOBOS<FeatType,LabelType>::UpdateWeightVec_L1(const DataPoint<FeatType, LabelType> &x)
	{
		int featDim = x.Dim();
		for (int i = 0; i < featDim; i++)
		{
			//lazy update
			if (x[i] != 0)
			{
				//in this process, we obtained w_(t)', that needs to be updated by stochastic gradient
				int stepK = this->curIterNum - this->timeStamp[i];
				this->timeStamp[i] = this->curIterNum;

				double alpha =  stepK * this->eta * this->lambda;
				double tmp = std::abs(this->weightVec[i]) - alpha;
				this->weightVec[i] = Sgn(this->weightVec[i]) * std::max(0.0,tmp);
				
			}
		}
		double y = this->Predict(x);
		for (int i = 0; i < featDim; i++)
		{
			if (x[i] != 0)
			{
				//update the weight
				double gt_i = this->lossFunc->GetGradient(x,y,i);
				this->weightVec[i] -= this->eta * gt_i;
			}
		}
		
		//update bias term
		double gt_i = this->lossFunc->GetBiasGradient(x,y);
		this->weightVec[this->weightDim - 1] -= this->eta * gt_i;

		return y;
	}

	/**
	 *  UpdateWeightVec_L2S Use squre of L2 as the regularizatio term
	 *  r(w) = 0.5 * lambda * ||w||^2
	 *
	 * @tparam FeatType
	 * @tparam LabelType
	 * @Param:  x
	 * @Param:  y
	 */
	template <typename FeatType, typename LabelType>
	double FOBOS<FeatType,LabelType>::UpdateWeightVec_L2S(const DataPoint<FeatType, LabelType> &x)
	{
		int featDim = x.Dim();
		for (int i = 0; i < featDim; i++)
		{
			//lazy update
			if (x[i] != 0)
			{
				int stepK = this->curIterNum - this->timeStamp[i];
				this->timeStamp[i] = this->curIterNum;
				this->weightVec[i] /= std::pow(1 + this->lambda,stepK);
			}
		}

		double y = this->Predict(x);
		for (int i = 0; i < featDim; i++)
		{
			if (x[i] != 0)
			{
				//update the weight
				double gt_i = this->lossFunc->GetGradient(x,y,i);
				this->weightVec[i] -= this->eta * gt_i;
			}
		}
		//update bias term
		double gt_i = this->lossFunc->GetBiasGradient(x,y);
		this->weightVec[this->weightDim - 1] -= this->eta * gt_i;
		return y;
	}

	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void FOBOS<FeatType, LabelType>::Reset()
	{
		Optimizer<FeatType, LabelType>::Reset();
		//reset time stamp
		memset(this->timeStamp,0,sizeof(size_t) * this->weightDim);
	}

	//called when a pass ends
	template <typename FeatType, typename LabelType>
	void FOBOS<FeatType,LabelType>::PassEnd()
	{
	}

	//called when a pass ends
	template <typename FeatType, typename LabelType>
	void FOBOS<FeatType,LabelType>::RoundEnd()
	{
		//here we need to bound the weights again
		int featDim = this->weightDim - 1;
		switch(normType)
		{
			case NormType_L1:
				for (int i = 0; i < featDim; i++)
				{
					int stepK = this->curIterNum - this->timeStamp[i];
					double tmp = std::abs(this->weightVec[i]) - stepK * this->eta * this->lambda;
					this->weightVec[i] = Sgn(this->weightVec[i]) * std::max(0.0,tmp);
				}
				break;
			case NormType_L2S:
				for (int i = 0; i < featDim; i++)
				{
					//truncatedd gradient
					int stepK = (this->curIterNum - this->timeStamp[i]); 
					double denorm = std::pow(1 + this->lambda,stepK);
					this->weightVec[i] /= denorm;
				}
				break;
			default:
				throw invalid_argument("Unsupported update rule!");
				break;
		}
		
	}

	//Change the dimension of weights
	template <typename FeatType, typename LabelType>
	void FOBOS<FeatType, LabelType>::UpdateWeightSize(int newDim)
	{
		if (newDim <= this->weightDim - 1)
			return;
		else
		{
			size_t* newT = new size_t[newDim + 1];
			memset(newT,0,sizeof(size_t) * (newDim + 1));
			memcpy(newT,this->timeStamp,sizeof(size_t) * (this->weightDim - 1));
			newT[newDim] = this->timeStamp[this->weightDim - 1];
			delete []this->timeStamp;
			this->timeStamp = newT;

			Optimizer<FeatType,LabelType>::UpdateWeightSize(newDim);
		}
	}
}


