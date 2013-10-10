/*************************************************************************
> File Name: RDA_L1.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 8/19/2013 Monday 1:49:43 PM
> Functions: Enhanced L1-RDA method
> Reference:
Xiao L. Dual averaging methods for regularized stochastic learning 
and online optimization[J]. The Journal of Machine Learning Research, 
2010, 9999: 2543-2596.
************************************************************************/

#pragma once

#include "Optimizer.h"
#include "../common/util.h"
#include <cmath>

namespace SOL
{
	template <typename FeatType, typename LabelType>
	class RDA_L1: public Optimizer<FeatType, LabelType>
	{
	public:
		RDA_L1(DataSet<FeatType, LabelType> &dataset, LossFunction<FeatType, LabelType> &lossFunc, bool enchance = true);
		~RDA_L1();

	public:
		void SetParameterEx(double lambda, double rou = -1);
        //try and get the best parameter
        virtual void BestParameter(); 
	
	protected:
		//this is the core of different updating algorithms
		//return the predict
		virtual double UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
		//reset the optimizer to this initialization
		virtual void BeginTrain();
		//called when a train ends
		virtual void EndTrain();
        //Change the dimension of weights
		virtual void UpdateWeightSize(int newDim);

	protected:
		double rou;

		double * gtVec; //average gradient vector

		bool is_enhanced;

	};

	template <typename FeatType, typename LabelType>
	RDA_L1<FeatType, LabelType>::RDA_L1(DataSet<FeatType, LabelType> &dataset, LossFunction<FeatType, LabelType> &lossFunc, bool enchance):
	Optimizer<FeatType, LabelType>(dataset, lossFunc) 
	{
        this->id_str = "RDA";
		this->rou = init_rou;
		this->gtVec = new double[this->weightDim];
		this->is_enhanced = enchance;
	}


	template <typename FeatType, typename LabelType>
	RDA_L1<FeatType,LabelType>::~RDA_L1()
	{
		if (this->gtVec != NULL)
			delete []this->gtVec;
	}

	template <typename FeatType, typename LabelType>
	double RDA_L1<FeatType,LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x)
	{
		size_t featDim = x.indexes.size();
		int index_i = 0;
		//obtain w_t
		double coeff = -this->eta / std::sqrt(this->curIterNum - 1);
		double lambda_t = this->lambda * (this->curIterNum - 1);
        if (this->is_enhanced == true){
            lambda_t += this->rou * std::sqrt(this->curIterNum - 1) / this->eta;
        }

		for (size_t i = 0; i < featDim; i++)
		{
			index_i = x.indexes[i];
			if (std::abs(this->gtVec[index_i]) > lambda_t)
				this->weightVec[index_i] = coeff * (this->gtVec[index_i] - lambda_t * Sgn(this->gtVec[index_i]));
			else
				this->weightVec[index_i] = 0;
		}

		//predict
		double y = this->Predict(x);
        double gt_i = this->lossFunc->GetGradient(x,y);

		//update the coeffs
		for (size_t i = 0; i < featDim; i++)
            this->gtVec[x.indexes[i]] += gt_i * x.features[i];

		//bias term
		this->gtVec[0] += gt_i;
		this->weightVec[0] = -this->eta  * this->gtVec[0] / std::sqrt((double)this->curIterNum);

        return y;
	}

	//
	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void RDA_L1<FeatType, LabelType>::BeginTrain()
	{
		Optimizer<FeatType, LabelType>::BeginTrain();
		memset(this->gtVec, 0, sizeof(double) * this->weightDim);
	}

	//called when a train ends
	template <typename FeatType, typename LabelType>
	void RDA_L1<FeatType, LabelType>::EndTrain()
	{
		if (this->curIterNum == 0)
			return;
		double coeff = -this->eta / std::sqrt((double)this->curIterNum);
		double lambda_t = this->lambda * this->curIterNum;
        if (this->is_enhanced == true){
            lambda_t += this->rou * std::sqrt(this->curIterNum) / this->eta;
        }

		for (int index_i = 1; index_i < this->weightDim; index_i++)
		{
			if (std::abs(this->gtVec[index_i]) > lambda_t)
				this->weightVec[index_i] = coeff * (this->gtVec[index_i] - lambda_t * Sgn(this->gtVec[index_i]));
			else
				this->weightVec[index_i] = 0;
		}
		
		Optimizer<FeatType, LabelType>::EndTrain();
	}

	template <typename FeatType, typename LabelType>
	void RDA_L1<FeatType,LabelType>::SetParameterEx(double lambda, double rou)
	{
		this->rou = rou >= 0 ? rou : this->rou;
		this->lambda = lambda >= 0 ? lambda : this->lambda;
	}

	//Change the dimension of weights
	template <typename FeatType, typename LabelType>
	void RDA_L1<FeatType, LabelType>::UpdateWeightSize(int newDim)
	{
		if (newDim < this->weightDim)
			return;
		else
		{
            newDim++;
			double* newT = new double[newDim];
            //copy info
			memcpy(newT,this->gtVec,sizeof(double) * this->weightDim);
            //set the rest to zero
			memset(newT + this->weightDim,0,sizeof(double) * (newDim - this->weightDim));
			delete []this->gtVec;
			this->gtVec = newT;

			Optimizer<FeatType,LabelType>::UpdateWeightSize(newDim - 1);
		}
	}

	//try and get the best parameter
	template <typename FeatType, typename LabelType>
	void RDA_L1<FeatType, LabelType>::BestParameter()
	{
        Optimizer<FeatType,LabelType>::BestParameter();
	}
}
