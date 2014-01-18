/************************************************************************
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
#include "../utils/util.h"
#include <cmath>

namespace SOL {
	template <typename FeatType, typename LabelType>
	class RDA_L1: public Optimizer<FeatType, LabelType> {
	public:
		RDA_L1(DataSet<FeatType, LabelType> &dataset, LossFunction<FeatType, LabelType> &lossFunc, bool enchance = true);
		~RDA_L1();

	public:
		void SetParameterEx( float gamma_rou = -1);
		/**
		* PrintOptInfo print the info of optimization algorithm
		*/
		virtual void PrintOptInfo() const {
			printf("--------------------------------------------------\n");
			printf("Algorithm: %s\n\n",this->Id_Str().c_str());
			printf("Learning Rate: %g\n", this->eta0);
			printf("Power t : 0.5\n");
			if (this->gamma_rou > 0){
				printf("gamma rou: %g\n", this->gamma_rou);
			}
			printf("lambda	: %g\n\n",this->lambda);
		}

	protected:
		//this is the core of different updating algorithms
		//return the predict
		virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
		//reset the optimizer to this initialization
		virtual void BeginTrain();
		//called when a train ends
		virtual void EndTrain();
		//Change the dimension of weights
		virtual void UpdateWeightSize(IndexType newDim);

	protected:
		float gamma_rou;
		s_array<float> gtVec; //average gradient vector

	};

	template <typename FeatType, typename LabelType>
	RDA_L1<FeatType, LabelType>::RDA_L1(DataSet<FeatType, LabelType> &dataset, LossFunction<FeatType, LabelType> &lossFunc, bool enchance):
		Optimizer<FeatType, LabelType>(dataset, lossFunc) {
			if(enchance == true){
				this->id_str = "enhanced RDA";
				this->gamma_rou = init_gammarou;
			}
			else{
				this->id_str = "RDA";
				this->gamma_rou = 0;
			}
			this->gtVec.resize(this->weightDim);
			//initail_t should be no less than 1,for the safety of update at the first step
			this->initial_t = this->initial_t < 1 ? 1 : this->initial_t;
	}


	template <typename FeatType, typename LabelType>
	RDA_L1<FeatType,LabelType>::~RDA_L1() {
	}

	template <typename FeatType, typename LabelType>
	float RDA_L1<FeatType,LabelType>::UpdateWeightVec(
		const DataPoint<FeatType, LabelType> &x) {
			float eta_coeff_time = pEta_sqrt(this->curIterNum, this->power_t);
			this->eta = this->eta0 / eta_coeff_time;

			size_t featDim = x.indexes.size();
			IndexType index_i = 0;
			//obtain w_t
			float lambda_t = this->lambda * this->curIterNum;
			if (this->gamma_rou > 0){
				lambda_t += this->gamma_rou * eta_coeff_time;
			}

			for (size_t i = 0; i < featDim; i++) {
				index_i = x.indexes[i];
				this->weightVec[index_i] = -this->eta * 
					trunc_weight(this->gtVec[index_i],lambda_t);
			}
			//bias
			this->weightVec[0] = -this->eta * this->gtVec[0];

			//predict
			float y = this->Predict(x);
			float gt_i = this->lossFunc->GetGradient(x.label,y);

			//update the coeffs
			for (size_t i = 0; i < featDim; i++)
				this->gtVec[x.indexes[i]] += gt_i * x.features[i];
			//bias term
			this->gtVec[0] += gt_i;

			return y;
	}

	//
	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void RDA_L1<FeatType, LabelType>::BeginTrain() {
		Optimizer<FeatType, LabelType>::BeginTrain();
		this->gtVec.zeros();
	}

	//called when a train ends
	template <typename FeatType, typename LabelType>
	void RDA_L1<FeatType, LabelType>::EndTrain() {
		Optimizer<FeatType, LabelType>::EndTrain();
	}

	template <typename FeatType, typename LabelType>
	void RDA_L1<FeatType,LabelType>::SetParameterEx( float gammarou) {
		this->gamma_rou = gammarou >= 0 ? gammarou : this->gamma_rou;
	}

	//Change the dimension of weights
	template <typename FeatType, typename LabelType>
	void RDA_L1<FeatType, LabelType>::UpdateWeightSize(IndexType newDim) {
		if (newDim < this->weightDim)
			return;
		else {
			this->gtVec.reserve(newDim + 1);
			this->gtVec.resize(newDim + 1);
			this->gtVec.zeros(this->gtVec.begin + this->weightDim, 
				this->gtVec.end);

			Optimizer<FeatType,LabelType>::UpdateWeightSize(newDim);
		}
	}
}
