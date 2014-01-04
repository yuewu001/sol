/*************************************************************************
> File Name: Ada_FOBOS.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: Sat 26 Oct 2013 12:17:04 PM SGT
> Descriptions: adaptive fobos algorithm
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

namespace SOL {
	template <typename FeatType, typename LabelType>
	class Ada_FOBOS: public Optimizer<FeatType, LabelType> {
	protected:
		float delta;
		s_array<size_t> timeStamp;
		s_array<float> s;
		s_array<float> u_t;

	public:
		Ada_FOBOS(DataSet<FeatType, LabelType> &dataset, 
			LossFunction<FeatType, LabelType> &lossFunc);
		~Ada_FOBOS();

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
	Ada_FOBOS<FeatType, LabelType>::Ada_FOBOS(DataSet<FeatType, LabelType> &dataset, 
		LossFunction<FeatType, LabelType> &lossFunc):
	Optimizer<FeatType, LabelType>(dataset, lossFunc){
		this->delta = init_delta;;
		this->timeStamp.resize(this->weightDim);
		this->s.resize(this->weightDim);
		this->u_t.resize(this->weightDim);

		this->id_str = "Adaptive FOBOS";
	}

	template <typename FeatType, typename LabelType>
	Ada_FOBOS<FeatType, LabelType>::~Ada_FOBOS() {
	}
	//update witt Composite Mirror-Descent
	template <typename FeatType, typename LabelType>
	float Ada_FOBOS<FeatType,LabelType>::UpdateWeightVec(
		const DataPoint<FeatType, LabelType> &x) {
			size_t featDim = x.indexes.size();
			IndexType index_i = 0;
			float alpha = this->eta0 * this->lambda;
			for (size_t i = 0; i < featDim; i++) {
				index_i = x.indexes[i];
				//update s[i]
				float Ht0i = this->delta + s[index_i];

				//to obtain w_(t + 1),i, first calculate w_t,i
				this->weightVec[index_i] = trunc_weight(this->weightVec[index_i],
					alpha * (this->curIterNum - this->timeStamp[index_i]) / Ht0i);

				//update the time stamp
				this->timeStamp[index_i] = this->curIterNum;
			}
			float y = this->Predict(x);
			//get gradient
			float gt = this->lossFunc->GetGradient(x.label,y);
			float gt_i = 0;

			//update s[i]
			for (size_t i = 0; i < featDim; i++) {
				index_i = x.indexes[i];
				gt_i = gt * x.features[i];

				this->s[index_i] = sqrt(s[index_i] * s[index_i] + gt_i * gt_i);
				float Htii = this->delta + s[index_i];
				//obtain w_(t + 1),i
				this->weightVec[index_i] -= this->eta0 * gt_i / Htii;
			}

			//bias term
			this->s[0] = sqrt(s[0] * s[0] + gt * gt);
			float Htii = this->delta + s[0];
			this->weightVec[0] -= this->eta0 * gt / Htii;

			return y;
	}

	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void Ada_FOBOS<FeatType, LabelType>::BeginTrain() {
		Optimizer<FeatType, LabelType>::BeginTrain();
		//reset time stamp
		this->timeStamp.zeros();
		this->s.zeros();
		this->u_t.zeros();
	}

	//called when a train ends
	template <typename FeatType, typename LabelType>
	void Ada_FOBOS<FeatType, LabelType>::EndTrain() {
		size_t iterNum = this->curIterNum + 1;
		float alpha = 0;
		for (IndexType index_i = 1; index_i < this->weightDim; index_i++) {
			//update s[i]
			float Ht0i = this->delta + s[index_i];
			alpha = this->lambda * this->eta0 * (iterNum - this->timeStamp[index_i]) / Ht0i;
			this->weightVec[index_i] = trunc_weight(this->weightVec[index_i], alpha);
		}

		Optimizer<FeatType,LabelType>::EndTrain();
	}

	//get the best model parameter
	template <typename FeatType, typename LabelType>
	void Ada_FOBOS<FeatType, LabelType>::BestParameter() {
		//first learn the best learning rate
		Optimizer<FeatType,LabelType>::BestParameter();
		float prevLambda = this->lambda;
		this->lambda = 0;

		//Select the best eta0
		float min_errorRate = 1;
		float bestDelta = 1;

		for (float delt = init_delta_min; delt <= init_delta_max; delt *= init_delta_step) {
			cout<<"delta= "<<delt<<"\n";
			this->delta = delt;
			float errorRate(0);
			errorRate = this->Train();

			if (errorRate < min_errorRate) {
				bestDelta = delt;
				min_errorRate = errorRate;
			}
			cout<<" mistake rate: "<<errorRate * 100<<" %\n";
		}

		this->delta = bestDelta;
		this->lambda = prevLambda;
		cout<<"Best Parameter:\tdelta = "<<this->delta<<"\n\n";
	}

	//set parameters for specific optimizers
	template <typename FeatType, typename LabelType>
	void Ada_FOBOS<FeatType, LabelType>::SetParameterEx(float delta) {
		this->delta = delta >= 0 ? delta : this->delta;
	}

	//Change the dimension of weights
	template <typename FeatType, typename LabelType>
	void Ada_FOBOS<FeatType, LabelType>::UpdateWeightSize(IndexType newDim) {
		if (newDim < this->weightDim)
			return;
		else {
			this->timeStamp.reserve(newDim + 1);
			this->timeStamp.resize(newDim + 1);
			//set the rest to zero
			this->timeStamp.zeros(this->timeStamp.begin + this->weightDim,
				this->timeStamp.end);

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

			Optimizer<FeatType,LabelType>::UpdateWeightSize(newDim);
		}
	}
}

