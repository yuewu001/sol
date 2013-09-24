/*************************************************************************
> File Name: ASM_L1.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/18 星期日 17:25:54
> Functions: Adaptive Subgradient Method for Online Learning and 
			Stachastic Optimization
> Reference:
		Duchi J, Hazan E, Singer Y. Adaptive subgradient methods for 
		online learning and stochastic optimization[J]. The Journal of 
		Machine Learning Research, 2011, 999999: 2121-2159.

		This file implements the L1 regularization
 ************************************************************************/

#pragma once
#include "Optimizer.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace SOL
{
	/**
	 *  Update rule: Primal Subgradient update and 
	 *  Composite Mirror-Descent Update
	 */
	enum ASM_UpdateRule
	{
		ASM_Update_PSU,
		ASM_Update_CMDU,
	};

	template <typename FeatType, typename LabelType>
	class ASM_L1: public Optimizer<FeatType, LabelType>
	{
	public:
		ASM_L1(DataSet<FeatType, LabelType> &dataset, 
			LossFunction<FeatType, LabelType> &lossFunc, ASM_UpdateRule updateRule = ASM_Update_PSU);
		~ASM_L1();

	public:
		//set parameters for specific optimizers
		void SetParameterEx(double lambda = -1, double delta = -1, double eta = -1);

		//select the best parameters for the model
		virtual void BestParameter();

	public:
		//Change the dimension of weights
		virtual void UpdateWeightSize(int newDim);
		
	protected:
		//this is the core of different updating algorithms
		virtual double UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
		double UpdateWeightVec_PSU(const DataPoint<FeatType, LabelType> &x);
		double UpdateWeightVec_CMD(const DataPoint<FeatType, LabelType> &x);

		//reset
		virtual void Reset();
		//called when a pass ends
		virtual void PassEnd();
		//called when a round ended
		virtual void RoundEnd();

	protected:
		ASM_UpdateRule updateRule;

		double delta;
		size_t *timeStamp;
		double *s;
		double *u_t;
	};

	template <typename FeatType, typename LabelType>
	ASM_L1<FeatType, LabelType>::ASM_L1(DataSet<FeatType, LabelType> &dataset, 
		LossFunction<FeatType, LabelType> &lossFunc, ASM_UpdateRule updateRule):
	Optimizer<FeatType, LabelType>(dataset, lossFunc), timeStamp(NULL),
	s(NULL), u_t(NULL)
	{
		this->delta = 1;
		this->updateRule = updateRule;

		this->timeStamp = new size_t[this->weightDim];
		this->s = new double[this->weightDim];
		this->u_t = new double[this->weightDim];
		this->Reset();
	}

	template <typename FeatType, typename LabelType>
	ASM_L1<FeatType, LabelType>::~ASM_L1()
	{
		if(this->timeStamp != NULL)
			delete []this->timeStamp;
		if (this->s != NULL)
			delete []this->s;
		if (this->u_t != NULL)
			delete []this->u_t;
	}
	//this is the core of different updating algorithms
	template <typename FeatType, typename LabelType>
	double ASM_L1<FeatType,LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
	{
		switch(updateRule)
		{
			case ASM_Update_PSU:
				return this->UpdateWeightVec_PSU(x);
				break;
			case ASM_Update_CMDU:
				return this->UpdateWeightVec_CMD(x);
				break;
			default:
				throw invalid_argument("Unkonw update rule!");
				break;
		}
	}

	//update witt Primal-dual Subgradient Update
	template <typename FeatType, typename LabelType>
	double ASM_L1<FeatType,LabelType>::UpdateWeightVec_PSU(const DataPoint<FeatType, LabelType> &x)
	{
		int featDim = x.Dim();
		for (int i = 0; i < featDim; i++)
		{
			//lazy update
			if (x[i] != 0)
			{
				//update s[i]
				double Htii = this->delta + s[i];
				this->weightVec[i] =  Sgn(-u_t[i]) * this->eta / Htii * 
					std::max(0.0,std::abs(u_t[i]) - this->lambda * this->curIterNum);
			}
		}
		
		//predict 
		double y = this->Predict(x);
		//update
		for (int i = 0; i < featDim; i++)
		{
			//lazy update
			if (x[i] != 0)
			{
				//get gradient
				double gt_i = this->lossFunc->GetGradient(x,y,i);
				//update s[i]
				this->s[i] = sqrt(this->s[i] * this->s[i] + gt_i * gt_i);
				this->u_t[i] += gt_i;
			}
		}
		//bias term
		double gt_i = this->lossFunc->GetBiasGradient(x,y);
		this->s[this->weightDim - 1] = sqrt(s[this->weightDim - 1] * s[this->weightDim - 1] + gt_i * gt_i);
		this->u_t[this->weightDim - 1] += gt_i;
		double Htii = this->delta + s[this->weightDim - 1];
		this->weightVec[this->weightDim - 1] =  Sgn(-u_t[this->weightDim - 1]) * this->eta / Htii * 
			std::max(0.0,std::abs(u_t[this->weightDim - 1]) - this->lambda * this->curIterNum);

		return y;
	}

	
	//update witt Composite Mirror-Descent
	template <typename FeatType, typename LabelType>
	double ASM_L1<FeatType,LabelType>::UpdateWeightVec_CMD(const DataPoint<FeatType, LabelType> &x)
	{
		int featDim = x.Dim();
		for (int i = 0; i < featDim; i++)
		{
			//lazy update
			if (x[i] != 0)
			{
				//update s[i]
				double Ht0i = this->delta + s[i];

				//to obtain w_(t + 1),i, first calculate w_t,i
				double tmp = std::abs(this->weightVec[i]) - 
					this->lambda * this->eta * (this->curIterNum - this->timeStamp[i]) / Ht0i;
				this->weightVec[i] = Sgn(this->weightVec[i]) * std::max(0.0,tmp);
				//update the time stamp
				this->timeStamp[i] = this->curIterNum;
			}
		}
		double y = this->Predict(x);

		//update s[i]
		for (int i = 0; i < featDim; i++)
		{
			if (x[i] != 0)
			{
				//get gradient
				double gt_i = this->lossFunc->GetGradient(x,y,i);
				this->s[i] = sqrt(s[i] * s[i] + gt_i * gt_i);
				double Htii = this->delta + s[i];
				//obtain w_(t + 1),i
				this->weightVec[i] -= this->eta * gt_i / Htii;
			}
		}

		//bias term
		double gt_i = this->lossFunc->GetBiasGradient(x,y);
		//update s[i]
		this->s[this->weightDim - 1] = sqrt(s[this->weightDim - 1] * s[this->weightDim - 1] + gt_i * gt_i);
		double Htii = this->delta + s[this->weightDim - 1];
		//obtain w_t,i
		double tmp = this->weightVec[this->weightDim - 1] - this->eta * gt_i / Htii;
		this->weightVec[this->weightDim - 1] = Sgn(tmp) * std::max(std::abs(tmp) - this->lambda * this->eta / Htii,0.0);

		return y;
	}

	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void ASM_L1<FeatType, LabelType>::Reset()
	{
		Optimizer<FeatType, LabelType>::Reset();
		//reset time stamp
		memset(this->timeStamp,0,sizeof(size_t) * this->weightDim);
		memset(this->s,0,sizeof(double) * this->weightDim);
		memset(this->u_t, 0 ,sizeof(double) * this->weightDim);
	}

	//called when a pass ends
	template <typename FeatType, typename LabelType>
	void ASM_L1<FeatType,LabelType>::PassEnd()
	{
	}
	
	//called when a round ended
	template <typename FeatType, typename LabelType>
	void ASM_L1<FeatType,LabelType>::RoundEnd()
	{
		switch(updateRule)
		{
		case ASM_Update_PSU:
			{
				for (int i = 0; i < this->weightDim; i++)
				{
					double Htii = this->delta + s[i];
					this->weightVec[i] =  Sgn(-u_t[i]) * this->eta / Htii * 
						std::max(0.0,std::abs(u_t[i]) - this->lambda * (this->curIterNum - 1));
				}
			}
			break;
		case ASM_Update_CMDU:
			{
				for (int i = 0; i < this->weightDim ; i++)
				{
					double Ht0i = this->delta + this->s[i];
					double tmp = std::abs(this->weightVec[i]) - 
						this->lambda * this->eta * (this->curIterNum - this->timeStamp[i]) / Ht0i;
					this->weightVec[i] = Sgn(this->weightVec[i]) * std::max(0.0,tmp);
				}
			}
			break;
		default:
			throw invalid_argument("Unkonw update rule!");
			break;
		}
	}

	//get the best model parameter
	template <typename FeatType, typename LabelType>
	void ASM_L1<FeatType, LabelType>::BestParameter()
	{
		double prevLambda = this->lambda;
		this->lambda = 1;

		//Select the best eta
		double eta_min = 1e-10;
		double eta_max = 1;
		double delt_min = 0.1;
		double delt_max = 10;

		float min_errorRate = 1;
		double bestEta = 1;
		double bestDelta = 1;

		this->dataSet.RandomOrder();

		for (double eta_c = eta_min; eta_c<= eta_max; eta_c *= 10)
		{
			this->eta = eta_c;
			for (double delt = delt_min; delt <= delt_max; delt *= 10)
			{
				cout<<"eta = "<<eta_c<<" delta= "<<delt;
				this->delta = delt;
				float errorRate(0);
				//for(int k = 1; k < 10; k++)
				{
					errorRate += this->RunOnce();
				}
				//errorRate /= 10;

				if (errorRate < min_errorRate)
				{
					bestEta = eta_c;
					bestDelta = delt;
					min_errorRate = errorRate;
				}
				cout<<" mistake rate: "<<errorRate * 100<<" %\n";
			}
		}
		
		this->eta = bestEta;
		this->delta = bestDelta;
		this->lambda = prevLambda;
		cout<<"Best Parameter:\n\teta = "<<this->eta<<"\tdelta = "<<this->delta<<"\n\n";
	}

	//set parameters for specific optimizers
	template <typename FeatType, typename LabelType>
	void ASM_L1<FeatType, LabelType>::SetParameterEx(double lambda , double delta , double eta )
	{
		this->lambda = lambda >= 0 ? lambda : this->lambda;
		this->delta = delta >= 0 ? delta : this->delta;
		this->eta = eta > 0 ? eta : this->eta;
	}

	//Change the dimension of weights
	template <typename FeatType, typename LabelType>
	void ASM_L1<FeatType, LabelType>::UpdateWeightSize(int newDim)
	{
		if (newDim < this->weightDim - 1)
			return;
		else
		{
			size_t* newT = new size_t[newDim + 1];
			memset(newT,0,sizeof(size_t) * (newDim + 1));
			memcpy(newT,this->timeStamp,sizeof(size_t) * (this->weightDim - 1));
			newT[newDim] = this->timeStamp[this->weightDim - 1];
			delete []this->timeStamp;
			this->timeStamp = newT;

			double* newS = new double[newDim + 1];
			memset(newS,0,sizeof(double) * (newDim + 1));
			memcpy(newS,this->s,sizeof(double) * (this->weightDim - 1));
			newS[newDim] = this->s[this->weightDim - 1];
			delete []this->s;
			this->s = newS;

			double* newUt = new double[newDim + 1];
			memset(newUt,0,sizeof(double) * (newDim + 1));
			memcpy(newUt,this->u_t,sizeof(double) * (this->weightDim - 1));
			newUt[newDim] = this->u_t[this->weightDim - 1];
			delete []this->u_t;
			this->u_t = newUt;

			Optimizer<FeatType,LabelType>::UpdateWeightSize(newDim);
		}
	}
}


