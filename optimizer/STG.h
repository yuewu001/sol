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


#include "../util.h"
#include "Optimizer.h"
#include <cmath>
#include <limits>

namespace SOL
{
	template <typename FeatType, typename LabelType>
	class STG: public Optimizer<FeatType, LabelType>
	{
	public:
		STG(DataSet<FeatType, LabelType> &dataset, 
			LossFunction<FeatType, LabelType> &lossFunc);
		~STG();

	public:
		void SetParameterEx(double lambda = -1,int K = 10, 
			double eta = -1,double theta = -1);
	protected:
		//this is the core of different updating algorithms
		//return the predict
		virtual double UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
		//reset
		virtual void Reset();
		//called when a pass ends
		virtual void PassEnd(){}
		//called when a round ended
		virtual void RoundEnd(); 

	public:
		//Change the dimension of weights
		virtual void UpdateWeightSize(int newDim);
		
	protected:
		int K;
		double theta; //truncate threshold

	protected:
		size_t*timeStamp;
	};

	template <typename FeatType, typename LabelType>
	STG<FeatType, LabelType>::STG(DataSet<FeatType, LabelType> &dataset, 
		LossFunction<FeatType, LabelType> &lossFunc):
	Optimizer<FeatType, LabelType>(dataset, lossFunc) , timeStamp(NULL)
	{
		this->K = 1;
		this->theta = std::numeric_limits<double>::max();
		this->timeStamp = new size_t[this->weightDim];
		this->Reset();
	}

	template <typename FeatType, typename LabelType>
	STG<FeatType, LabelType>::~STG()
	{
		if(this->timeStamp != NULL)
			delete []this->timeStamp;
	}

	//this is the core of different updating algorithms
	//return the predict
	template <typename FeatType, typename LabelType>
	double STG<FeatType,LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
	{
		int featDim = x.Dim();
		for (int i = 0; i < featDim; i++)
		{
			//lazy update
			if (x[i] != 0)
			{
				//truncated gradient
				//in this process, we obtained w_(t)', that needs to be updated by stochastic gradient
				int stepK = ((this->curIterNum - this->timeStamp[i]) / this->K) * this->K;

				if (this->timeStamp[i] == 0)
				{
					this->timeStamp[i] = this->curIterNum;
					stepK = 0;
				}
				else
					this->timeStamp[i] += stepK;

				double alpha =  stepK * this->eta * this->lambda;
				double tmp = std::abs(this->weightVec[i]) - alpha;
				this->weightVec[i] = Sgn(this->weightVec[i]) * std::max(0.0,tmp);
			}
		}
		//predict
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

		//bias term
		double gt_i = this->lossFunc->GetBiasGradient(x,y);
		this->weightVec[this->weightDim -1] -= this->eta * gt_i;

		return y;
	}
	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void STG<FeatType, LabelType>::Reset()
	{
		Optimizer<FeatType, LabelType>::Reset();
		//reset time stamp
		memset(this->timeStamp,0,sizeof(size_t) * this->weightDim);
	}

	//called when a pass ends
	template <typename FeatType, typename LabelType>
	void STG<FeatType,LabelType>::RoundEnd()
	{
		//here we need to bound the weights again
		int featDim = this->weightDim - 1;
		for (int i = 0; i < featDim; i++)
		{
			//truncatedd gradient
			int stepK = ((this->curIterNum - this->timeStamp[i]) / this->K ) * this->K; 
			double tmp = std::abs(this->weightVec[i]) - stepK * this->eta * this->lambda;
			this->weightVec[i] = Sgn(this->weightVec[i]) * std::max(0.0,tmp);
		}
	}

	template <typename FeatType, typename LabelType>
	void STG<FeatType, LabelType>::SetParameterEx(double lambda , int k,
		double Eta,  double Theta) 
	{
		this->lambda  = lambda >= 0 ? lambda : this->lambda;
		this->theta = Theta > 0 ? Theta : this->theta;
		this->eta = Eta > 0 ? Eta : this->eta;
		this->K = k > 0 ? k : this->K;
	}

	//Change the dimension of weights
	template <typename FeatType, typename LabelType>
	void STG<FeatType, LabelType>::UpdateWeightSize(int newDim)
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


