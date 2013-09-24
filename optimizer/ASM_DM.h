/*************************************************************************
> File Name: STG.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/18 星期日 17:25:54
> Functions: Adaptive Subgradient Method for Online Learning and 
            Stachastic Optimization
> Reference:
        Duchi J, Hazan E, Singer Y. Adaptive subgradient methods for 
        online learning and stochastic optimization[J]. The Journal of 
        Machine Learning Research, 2011, 999999: 2121-2159.
 ************************************************************************/

#pragma once
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
		void SetParameter(double g = -1, double theta = -1, double eta = -1,
			int K = 10, double decaRate = 0.01);

		virtual void BestParameter();
	protected:
		void BestParameter_g(double max_errRate);

	protected:
		//this is the core of different updating algorithms
		virtual void UpdateWeightVec(const DataPoint<FeatType, LabelType> &x, double y);
		//reset
		virtual void Reset();
		//called when a pass ends
		virtual void PassEnd();

	protected:
		double eta; //learning rate

	protected:
		int K;
		double g; //sparsificaion rate
		double theta; //truncate threshold
		double allowDecayRate; //tolerable decay rate of accuracy

		size_t curIterNum;
		size_t *timeStamp;
	};

	template <typename FeatType, typename LabelType>
	STG<FeatType, LabelType>::STG(DataSet<FeatType, LabelType> &dataset, 
		LossFunction<FeatType, LabelType> &lossFunc):
	Optimizer<FeatType, LabelType>(dataset, lossFunc), timeStamp(NULL)
	{
		this->K = 10;
		this->curIterNum = 0;
		this->g = 0.1;
		this->eta = 1e-3;
		this->theta = numeric_limits<double>::max();
		this->allowDecayRate = 0.01;

		this->timeStamp = new size_t[this->weightDim];
		memset(this->timeStamp, 0, sizeof(size_t) * this->weightDim);
	}

	template <typename FeatType, typename LabelType>
	STG<FeatType, LabelType>::~STG()
	{
		if(this->timeStamp != NULL)
			delete []this->timeStamp;
	}

	template <typename FeatType, typename LabelType>
	void STG<FeatType,LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x, double y)
	{
		this->curIterNum++;

		if(this->lossFunc->IsCorrect(x,y) == true)
			return;

		for (int i = 0; i < this->weightDim; i++)
		{
			//lazy update
			if (x[i] != 0)
			{
				//update the weight
				double gt_i = this->lossFunc->GetGradient(x,y,i);
				this->weightVec[i] -= eta * gt_i;

				//truncatedd gradient
				int stepK = ((this->curIterNum - this->timeStamp[i]) / this->K)  * this->K;
				double alpha = stepK * this->eta * this->g;

				if (this->weightVec[i] > 0 && this->weightVec[i] <= this->theta)
					this->weightVec[i] = std::max(0.0,this->weightVec[i] - alpha);
				else if (this->weightVec[i] >= -theta && this->weightVec[i] < 0)
					this->weightVec[i] = std::min(0.0,this->weightVec[i] + alpha);

				//update the time stamp
				this->timeStamp[i] += stepK;
			}
		}
	}
	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void STG<FeatType, LabelType>::Reset()
	{
		Optimizer<FeatType, LabelType>::Reset();
		//reset time stamp
		memset(this->timeStamp,0,sizeof(size_t) * this->weightDim);
		this->curIterNum = 0;
	}

	//called when a pass ends
	template <typename FeatType, typename LabelType>
	void STG<FeatType,LabelType>::PassEnd()
	{
	}


	//get the best model parameter
	template <typename FeatType, typename LabelType>
	void STG<FeatType, LabelType>::BestParameter()
	{
		//0. set sparsification to zero 
		this->g = 0;

		//1. Select the best eta
		double eta_min = 1e-10;
		double eta_max = 1;

		float min_errorRate = 1;
		double bestEta = 1;
		//random order
		this->dataSet.PreProcess(DataSetPreProcess_Rand); 

		for (double eta_c = eta_min; eta_c<= eta_max; eta_c *= 10)
		{
			this->eta = eta_c;
			float errorRate = this->RunOnce();
			if (errorRate < min_errorRate)
			{
				bestEta = eta_c;
				min_errorRate = errorRate;
			}
			cout<<"eta = "<<eta_c<<"\t mistake rate: "<<errorRate * 100<<" %\n";
		}
		this->eta = bestEta;
		cout<<"Best Parameter:\n\teta = "<<this->eta<<"\n\n";
		//2. Find the best g( sparsification rate)
		double bestAccu = 1 - min_errorRate;
		double max_errorRate = 1 - bestAccu * (1 - this->allowDecayRate);
		this->BestParameter_g(max_errorRate);
	}

	template <typename FeatType, typename LabelType>
	void STG<FeatType, LabelType>::BestParameter_g(double max_errRate)
	{
		double g_min = 1e-3;
		double g_max = 10;
		for (double tmp_g = g_min; tmp_g <= g_max; tmp_g *= 10)
		{
			this->g = tmp_g;
			float errRate = this->RunOnce();
			if (errRate > max_errRate)
			{
				this->g = tmp_g / 10;
				break;
			}
			cout<<"g= "<<tmp_g<<"\t mistake rate: "<<errRate * 100<<" %\n";
		}
		cout<<"\tSelected g= "<<this->g<<"\n";
	}

	template <typename FeatType, typename LabelType>
	void STG<FeatType, LabelType>::SetParameter(double g ,double theta , double eta, int K , double decaRate) 
	{
		this->g  = g > 0 ? g : this->g;
		this->theta = theta > 0 ? theta : this->theta;
		this->eta = eta > 0 ? eta : this->eta;
		this->K = K > 0 ? K : this->K;
		this->allowDecayRate = decaRate > 0 ? decaRate : this->allowDecayRate;
	}
}


