/*************************************************************************
> File Name: Ada_RDA.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/18 Sunday 17:25:54
> Functions: Adaptive RDA
> Reference:
Duchi J, Hazan E, Singer Y. Adaptive subgradient methods for 
online learning and stochastic optimization[J]. The Journal of 
Machine Learning Research, 2011, 999999: 2121-2159.

This file implements the L1 regularization
************************************************************************/

#ifndef HEADER_ADA_RDA
#define HEADER_ADA_RDA

#include "Optimizer.h"
#include <cmath>
#include <stdexcept>

namespace SOL {
	template <typename FeatType, typename LabelType>
	class Ada_RDA: public Optimizer<FeatType, LabelType> {

	protected:
		float delta;
		s_array<float> s;
		s_array<float> u_t;
	public:
		Ada_RDA(DataSet<FeatType, LabelType> &dataset, 
			LossFunction<FeatType, LabelType> &lossFunc);
		~Ada_RDA();

	public:
		//set parameters for specific optimizers
		void SetParameterEx(float delta = -1);

		//select the best parameters for the model
		virtual void BestParameter();
		/**
		* PrintOptInfo print the info of optimization algorithm
		*/
		virtual void PrintOptInfo() const {
			Optimizer<FeatType,LabelType>::PrintOptInfo();
			printf("delta : %g\n\n", this->delta);
		}

	protected:
		//this is the core of different updating algorithms
		virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);

		//Change the dimension of weights
		virtual void UpdateWeightSize(IndexType newDim);

		//reset
		virtual void BeginTrain();
		//called when a train ends
		virtual void EndTrain();


	};

	template <typename FeatType, typename LabelType>
	Ada_RDA<FeatType, LabelType>::Ada_RDA(DataSet<FeatType, LabelType> &dataset, 
		LossFunction<FeatType, LabelType> &lossFunc):
	Optimizer<FeatType, LabelType>(dataset, lossFunc){
		this->delta = init_delta;;
		this->s.resize(this->weightDim);
		this->u_t.resize(this->weightDim);

		this->id_str = "Adaptive RDA";
	}

	template <typename FeatType, typename LabelType>
	Ada_RDA<FeatType, LabelType>::~Ada_RDA() {
	}
	//this is the core of different updating algorithms
	template <typename FeatType, typename LabelType>
	float Ada_RDA<FeatType,LabelType>::UpdateWeightVec(
		const DataPoint<FeatType, LabelType> &x) {
			size_t featDim = x.indexes.size();
			IndexType index_i = 0;

			//obtain w_t
			for (size_t i = 0; i < featDim; i++) {
				index_i = x.indexes[i];
				//lazy update
				//update s[i]
				float Htii = this->delta + sqrtf(s[index_i]);
				this->weightVec[index_i] = -this->eta0 / Htii *
					trunc_weight(u_t[index_i], this->lambda * (this->curIterNum - 1));
			}

			//predict 
			float y = this->Predict(x);
			//get gradient
			float gt = this->lossFunc->GetGradient(x.label, y);
			if (gt != 0){
				float gt_i = 0;
				//update
				for (size_t i = 0; i < featDim; i++) {
					index_i = x.indexes[i];
					gt_i = gt * x.features[i];

					this->s[index_i] += gt_i * gt_i;
					this->u_t[index_i] += gt_i;
				}
				//bias term
				this->s[0] += gt * gt;
				this->u_t[0] += gt;
				float Htii = this->delta + sqrtf(s[0]);
				this->weightVec[0] = -u_t[0] * this->eta0 / Htii;
				this->update_times++;
			}
			return y;
	}

	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void Ada_RDA<FeatType, LabelType>::BeginTrain() {
		Optimizer<FeatType, LabelType>::BeginTrain();
		this->s.zeros();
		this->u_t.zeros();
	}
	//called when a train ends
	template <typename FeatType, typename LabelType>
	void Ada_RDA<FeatType, LabelType>::EndTrain() {
		Optimizer<FeatType, LabelType>::EndTrain();
	}

	//get the best model parameter
	template <typename FeatType, typename LabelType>
	void Ada_RDA<FeatType, LabelType>::BestParameter() {
		//first learn the best learning rate
		Optimizer<FeatType, LabelType>::BestParameter();
		float prevLambda = this->lambda;
		this->lambda = 0;

		//Select the best eta0
		float min_errorRate = 1;
		float bestDelta = 1;

		for (float delt = init_delta_min; delt <= init_delta_max; delt *= init_delta_step) {
			cout << "delta= " << delt << "\n";
			this->delta = delt;
			float errorRate(0);
			errorRate = this->Train();

			if (errorRate < min_errorRate) {
				bestDelta = delt;
				min_errorRate = errorRate;
			}
			cout << " mistake rate: " << errorRate * 100 << " %\n";
		}

		this->delta = bestDelta;
		this->lambda = prevLambda;
		cout << "Best Parameter:\tdelta = " << this->delta << "\n\n";
	}

	//set parameters for specific optimizers
	template <typename FeatType, typename LabelType>
	void Ada_RDA<FeatType, LabelType>::SetParameterEx(float delta) {
		this->delta = delta > 0 ? delta : this->delta;
	}

	//Change the dimension of weights
	template <typename FeatType, typename LabelType>
	void Ada_RDA<FeatType, LabelType>::UpdateWeightSize(IndexType newDim) {
		if (newDim < this->weightDim)
			return;
		else {
			this->s.reserve(newDim + 1);
			this->s.resize(newDim + 1);
			//set the rest to zero
			this->s.zeros(this->s.begin + this->weightDim,
				this->s.end);

			this->u_t.reserve(newDim + 1);
			this->u_t.resize(newDim + 1);
			//set the rest to zero
			this->u_t.zeros(this->u_t.begin + this->weightDim,
				this->u_t.end);

			Optimizer<FeatType, LabelType>::UpdateWeightSize(newDim);
		}
	}
}
#endif
