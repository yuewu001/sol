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
#include "../common/global.h"
#include "../common/util.h"

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

	protected:
		//this is the core of different updating algorithms
		virtual double UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
		double UpdateWeightVec_L1(const DataPoint<FeatType, LabelType> &x);
		double UpdateWeightVec_L2S(const DataPoint<FeatType, LabelType> &x);

        //Change the dimension of weights
		virtual void UpdateWeightSize(int newDim);

		//reset
		virtual void BeginTrain();

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
		int featDim = x.indexes.size();
        int index_i = 0;
		for (int i = 0; i < featDim; i++)
		{
            index_i = x.indexes[i];
			//lazy update
            
            //in this process, we obtained w_(t)', that needs to be updated by stochastic gradient
            int stepK = this->curIterNum - this->timeStamp[index_i];
            this->timeStamp[index_i] = this->curIterNum;

            double alpha =  stepK * this->eta * this->lambda;
            double tmp = std::abs(this->weightVec[index_i]) - alpha;
            this->weightVec[index_i] = Sgn(this->weightVec[index_i]) * std::max(0.0,tmp);
		}

		double y = this->Predict(x);
        double gt_i = this->lossFunc->GetGradient(x,y);
		for (int i = 0; i < featDim; i++)
            this->weightVec[x.indexes[i]] -= this->eta * gt_i * x.features[i];//update the weight
		
		//update bias term
		this->weightVec[0] -= this->eta * gt_i;

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
		int featDim = x.indexes.size();
        int index_i = 0;
		for (int i = 0; i < featDim; i++)
        {
            //lazy update
            index_i = x.indexes[i];
            int stepK = this->curIterNum - this->timeStamp[index_i];
            this->timeStamp[index_i] = this->curIterNum;
            this->weightVec[index_i] /= std::pow(1 + this->lambda,stepK);
        }

		double y = this->Predict(x);
        double gt_i = this->lossFunc->GetGradient(x,y);

		for (int i = 0; i < featDim; i++)
            this->weightVec[x.indexes[i]] -= this->eta * gt_i * x.features[i];		//update the weight

		//update bias term
		this->weightVec[0] -= this->eta * gt_i;
		return y;
	}

	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void FOBOS<FeatType, LabelType>::BeginTrain()
	{
		Optimizer<FeatType, LabelType>::BeginTrain();
		//reset time stamp
		memset(this->timeStamp,0,sizeof(size_t) * this->weightDim);
	}

	//Change the dimension of weights
	template <typename FeatType, typename LabelType>
	void FOBOS<FeatType, LabelType>::UpdateWeightSize(int newDim)
	{
		if (newDim < this->weightDim)
			return;
		else
		{
            newDim++;
			size_t* newT = new size_t[newDim];
            //copy info
			memcpy(newT,this->timeStamp,sizeof(size_t) * this->weightDim);
            //set the rest to zero
			memset(newT + this->weightDim,0,sizeof(size_t) * (newDim - this->weightDim));
			delete []this->timeStamp;
			this->timeStamp = newT;

			Optimizer<FeatType,LabelType>::UpdateWeightSize(newDim - 1);
		}
	}
}

