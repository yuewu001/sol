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
		void SetParameterEx(double lambda, double gamma = -1, double rou = -1);
        //try and get the best parameter
        virtual void BestParameter(); 
	
	protected:
		//this is the core of different updating algorithms
		//return the predict
		virtual double UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
		//standard RDA
		virtual double UpdateWeightVec_S(const DataPoint<FeatType, LabelType> &x);
		//enhanced RDA
		virtual double UpdateWeightVec_E(const DataPoint<FeatType, LabelType> &x);
		//reset the optimizer to this initialization
		virtual void BeginTrain();
		//called when a train ends
		virtual void EndTrain();
        //Change the dimension of weights
		virtual void UpdateWeightSize(int newDim);

	protected:
		double rou;
		double gamma;

		double * gtVec; //average gradient vector

		bool is_enchanced;

	};

	template <typename FeatType, typename LabelType>
	RDA_L1<FeatType, LabelType>::RDA_L1(DataSet<FeatType, LabelType> &dataset, LossFunction<FeatType, LabelType> &lossFunc, bool enchance):
	Optimizer<FeatType, LabelType>(dataset, lossFunc) 
	{
        this->id_str = "RDA";
		this->gamma = init_gamma;
		this->rou = init_rou;
		this->gtVec = new double[this->weightDim];
		this->is_enchanced = enchance;
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
		switch(this->is_enchanced)
		{
		case true:
			return this->UpdateWeightVec_E(x);
			break;
		case false:
			return this->UpdateWeightVec_S(x);
			break;
		}
		return 0;
	}
	//standard RDA
	template <typename FeatType, typename LabelType>
	double RDA_L1<FeatType,LabelType>::UpdateWeightVec_S(const DataPoint<FeatType, LabelType> &x)
	{
		size_t featDim = x.indexes.size();
		int index_i = 0;
		//obtain w_t
		double coeff = -this->eta / std::sqrt(this->curIterNum - 1);
		double lambda_t = this->lambda * (this->curIterNum - 1);
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
	//enhanced RDA
	template <typename FeatType, typename LabelType>
	double RDA_L1<FeatType,LabelType>::UpdateWeightVec_E(const DataPoint<FeatType, LabelType> &x)
	{
		double y = this->Predict(x);

		//update average gradient
		double coeff1 = 1.0 / this->curIterNum;
		double coeff2 = 1.0 - coeff1;
		double coeff3 = std::sqrt((double)this->curIterNum);
		double lambda_t = this->lambda + this->gamma * this->rou / coeff3;

        double gt_i = this->lossFunc->GetGradient(x,y);

		int featDim = x.indexes.size();

        for (int i = 1; i < this->weightDim; i++)
            this->gtVec[i] *= coeff2;
		for (int i = 0; i < featDim; i++)
        {
            this->gtVec[x.indexes[i]] += coeff1 * gt_i * x.features[i];
        }
        for  (int i = 1; i < this->weightDim; i++)
        {
			//update weights
			if (this->gtVec[i] <= lambda_t && this->gtVec[i] >= -lambda_t)
				this->weightVec[i] = 0;
			else
				this->weightVec[i] = (- coeff3 / this->gamma) * 
					(this->gtVec[i] - lambda_t * Sgn(this->gtVec[i]));
		}

		//bias term
		this->gtVec[0] = coeff2 * this->gtVec[0] + coeff1 * gt_i;
		this->weightVec[0] = (-coeff3 / this->gamma) * this->gtVec[0];

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
		if (this->is_enchanced == true || this->curIterNum == 0)
			return;
		double coeff = -this->eta / std::sqrt((double)this->curIterNum);
		double lambda_t = this->lambda * this->curIterNum;

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
	void RDA_L1<FeatType,LabelType>::SetParameterEx(double lambda,double gamma, double rou)
	{
		this->gamma = gamma > 0 ? gamma : this->gamma;
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
		if (this->is_enchanced == false)
			Optimizer<FeatType,LabelType>::BestParameter();
	}
}
