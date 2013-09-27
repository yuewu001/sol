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

				
	protected:
		//this is the core of different updating algorithms
		virtual double UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
		double UpdateWeightVec_PSU(const DataPoint<FeatType, LabelType> &x);
		double UpdateWeightVec_CMD(const DataPoint<FeatType, LabelType> &x);

        //Change the dimension of weights
		virtual void UpdateWeightSize(int newDim);

		//reset
		virtual void BeginTrain();

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
		int featDim = x.indexes.size();
        int index_i = 0;
		for (int i = 0; i < featDim; i++)
        {
            index_i = x.indexes[i];
            //lazy update
            //update s[i]
            double Htii = this->delta + s[index_i];
            this->weightVec[index_i] =  Sgn(-u_t[index_i]) * this->eta / Htii * 
                std::max(0.0,std::abs(u_t[index_i]) - this->lambda * this->curIterNum);
        }
		
		//predict 
		double y = this->Predict(x);
        //get gradient
        double gt = this->lossFunc->GetGradient(x,y);

        double gt_i = 0;
		//update
		for (int i = 0; i < featDim; i++)
		{
            index_i = x.indexes[i];
            gt_i = gt * x.features[i];

            this->s[index_i] = sqrt(this->s[index_i] * this->s[index_i] + gt_i * gt_i);
            this->u_t[index_i] += gt_i;
		}
		//bias term
		this->s[0] = sqrt(s[0] * s[0] + gt * gt);
		this->u_t[0] += gt;
		double Htii = this->delta + s[0];
		this->weightVec[0] =  Sgn(-u_t[0]) * this->eta / Htii * 
			std::max(0.0,std::abs(u_t[0]) - this->lambda * this->curIterNum);

		return y;
	}

	
	//update witt Composite Mirror-Descent
	template <typename FeatType, typename LabelType>
	double ASM_L1<FeatType,LabelType>::UpdateWeightVec_CMD(const DataPoint<FeatType, LabelType> &x)
	{
		int featDim = x.indexes.size();
        int index_i = 0;
		for (int i = 0; i < featDim; i++)
        {
            index_i = x.indexes[i];
            //update s[i]
            double Ht0i = this->delta + s[index_i];

            //to obtain w_(t + 1),i, first calculate w_t,i
            double tmp = std::abs(this->weightVec[index_i]) - 
                this->lambda * this->eta * (this->curIterNum - this->timeStamp[index_i]) / Ht0i;
            this->weightVec[index_i] = Sgn(this->weightVec[index_i]) * std::max(0.0,tmp);
            //update the time stamp
            this->timeStamp[index_i] = this->curIterNum;
        }
		double y = this->Predict(x);
        //get gradient
        double gt = this->lossFunc->GetGradient(x,y);
        double gt_i = 0;

		//update s[i]
		for (int i = 0; i < featDim; i++)
		{
            index_i = x.indexes[i];
            gt_i = gt * x.features[i];

            this->s[index_i] = sqrt(s[index_i] * s[index_i] + gt_i * gt_i);
            double Htii = this->delta + s[index_i];
            //obtain w_(t + 1),i
            this->weightVec[index_i] -= this->eta * gt_i / Htii;
		}

		//bias term
		//update s[i]
		this->s[0] = sqrt(s[0] * s[0] + gt * gt);
		double Htii = this->delta + s[0];
		//obtain w_t,i
		double tmp = this->weightVec[0] - this->eta * gt / Htii;
		this->weightVec[0] = Sgn(tmp) * std::max(std::abs(tmp) - this->lambda * this->eta / Htii,0.0);

		return y;
	}

	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void ASM_L1<FeatType, LabelType>::BeginTrain()
	{
		Optimizer<FeatType, LabelType>::BeginTrain();
		//reset time stamp
		memset(this->timeStamp,0,sizeof(size_t) * this->weightDim);
		memset(this->s,0,sizeof(double) * this->weightDim);
		memset(this->u_t, 0 ,sizeof(double) * this->weightDim);
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

		for (double eta_c = eta_min; eta_c<= eta_max; eta_c *= 10)
		{
			this->eta = eta_c;
			for (double delt = delt_min; delt <= delt_max; delt *= 10)
			{
				cout<<"eta = "<<eta_c<<" delta= "<<delt;
				this->delta = delt;
				float errorRate(0);
                errorRate += this->Train();

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
		if (newDim < this->weightDim)
			return;
		else
		{
            newDim++;
			size_t* newT = new size_t[newDim];
			double* newS = new double[newDim + 1];
			double* newUt = new double[newDim + 1];
            //copy info
			memcpy(newT,this->timeStamp,sizeof(size_t) * this->weightDim);
			memcpy(newS,this->s,sizeof(double) * this->weightDim);
			memcpy(newUt,this->u_t,sizeof(double) * this->weightDim);
            //set the rest to zero
			memset(newT + this->weightDim,0,sizeof(size_t) * (newDim - this->weightDim));
			memset(newS + this->weightDim,0,sizeof(double) * (newDim - this->weightDim));
			memset(newUt + this->weightDim,0,sizeof(double) * (newDim - this->weightDim));

			delete []this->timeStamp;
			delete []this->s;
			delete []this->u_t;

			this->timeStamp = newT;
			this->s = newS;
			this->u_t = newUt;

			Optimizer<FeatType,LabelType>::UpdateWeightSize(newDim - 1);
		}
	}
}


