/*************************************************************************
> File Name: Diagonal AROW
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/18 Sunday 17:25:54
> Functions: Diagonal Adaptive Regularization of Weight Vectors
> Reference: 
Crammer, Koby, Alex Kulesza, and Mark Dredze. "Adaptive regularization 
of weight vectors." Machine Learning (2009): 1-33.
************************************************************************/

#pragma once


#include "../utils/util.h"
#include "Optimizer.h"
#include "../loss/SquaredHingeLoss.h"
#include <cmath>
#include <limits>

namespace SOL {
	template <typename FeatType, typename LabelType>
	class SSAROW: public Optimizer<FeatType, LabelType> {
	protected:
		float r;
		s_array<float> sigma_w;
		s_array<size_t> timeStamp;
		s_array<float> sum_rate;

		size_t iter_num;


	public:
		SSAROW(DataSet<FeatType, LabelType> &dataset, 
			LossFunction<FeatType, LabelType> &lossFunc);
		virtual ~SSAROW();

	public:
		void SetParameterEx(float r = -1);
		//select the best parameters for the model
		virtual void BestParameter();
		/**
		* PrintOptInfo print the info of optimization algorithm
		*/
		virtual void PrintOptInfo() const {
			printf("--------------------------------------------------\n");
			printf("Algorithm: %s\n\n",this->Id_Str().c_str());
			printf("lambda:\t%g\n", this->lambda);
			printf("r:\t%g\n\n", this->r);
		}

	protected:
		//this is the core of different updating algorithms
		virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
		//reset the optimizer to this initialization
		virtual void BeginTrain();
		//called when a train ends
		virtual void EndTrain();

		//Change the dimension of weights
		virtual void UpdateWeightSize(IndexType newDim);
	};

	template <typename FeatType, typename LabelType>
	SSAROW<FeatType, LabelType>::SSAROW(DataSet<FeatType, LabelType> &dataset, 
		LossFunction<FeatType, LabelType> &lossFunc):
	Optimizer<FeatType, LabelType>(dataset, lossFunc) {
		this->id_str = "truncated gradient AROW";
		this->r = init_r;
		this->sigma_w.resize(this->weightDim);
		this->timeStamp.resize(this->weightDim);
		this->lossFunc = new SquaredHingeLoss<FeatType, LabelType>;
	}

	template <typename FeatType, typename LabelType>
	SSAROW<FeatType, LabelType>::~SSAROW() {
		if (this->lossFunc != NULL){
			delete this->lossFunc;
			this->lossFunc = NULL;
		}
	}

	//this is the core of different updating algorithms
	//return the predict
	template <typename FeatType, typename LabelType>
	float SSAROW<FeatType,LabelType>::UpdateWeightVec(
		const DataPoint<FeatType, LabelType> &x) {
			IndexType* p_index = x.indexes.begin;
			float* p_feat = x.features.begin;

			//obtain w_t
			float y = this->Predict(x); 
			float gt_i = this->lossFunc->GetGradient(x.label,y);
			//update w_t
			if(gt_i != 0){
				//calculate learning rate
				this->eta = this->r;
				while(p_index != x.indexes.end){
					this->eta += (*p_feat) * (*p_feat) * this->sigma_w[*p_index];
					p_index++;p_feat++;
				}

				this->eta = 0.5f / this->eta;
				this->sum_rate.push_back(this->sum_rate.last() + 
					this->eta * this->lambda);
				gt_i *= this->eta;

				float last_g_sum = this->sum_rate.last();
				p_index = x.indexes.begin;
				p_feat = x.features.begin;
				while(p_index != x.indexes.end){
					//update u_t
					this->weightVec[*p_index] -= gt_i *
						(*p_feat) * this->sigma_w[*p_index];  

					//L1 lazy update
					//size_t stepK = this->iter_num - this->timeStamp[*p_index];
					float gravity = last_g_sum - 
						this->sum_rate[this->timeStamp[*p_index]];
					//float gravity = stepK * this->lambda * this->beta_t / 2.f;
					this->timeStamp[*p_index] = this->iter_num;

					this->weightVec[*p_index]= 
						trunc_weight(this->weightVec[*p_index],
						gravity *(this->sigma_w[*p_index])); 

					//update sigma_w
					this->sigma_w[*p_index] *= this->r / (this->r + 
						this->sigma_w[*p_index] * (*p_feat) * (*p_feat));
					p_index++;p_feat++;
				}

				//bias term
				this->weightVec[0] -= gt_i * this->sigma_w[0];
				this->sigma_w[0] *= this->r / (this->r + this->sigma_w[0]);
				this->timeStamp[0] = this->iter_num;

				this->iter_num++;
			}
			return y;
	}

	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void SSAROW<FeatType, LabelType>::BeginTrain() {
		Optimizer<FeatType, LabelType>::BeginTrain();

		this->timeStamp.zeros();
		this->sigma_w.set_value(1);
		this->sum_rate.push_back(0);
		this->iter_num = 1; //force to begin from 1, as sum_rate depends on this value
        //this->is_normalize = true; //AROW requires normalization
	}

	//called when a train ends
	template <typename FeatType, typename LabelType>
	void SSAROW<FeatType, LabelType>::EndTrain() {
		float gravity = 0;
		//this->beta_t = 1.f / this->r;
		for (IndexType index_i = 1; index_i < this->weightDim; index_i++) {
			//L1 lazy update
			gravity = this->sum_rate.last() - this->sum_rate[this->timeStamp[index_i]];
			//size_t stepK = this->curIterNum - this->timeStamp[index_i];
			//gravity = stepK * this->lambda * this->beta_t / 2.f;

			//this->timeStamp[index_i] = this->curIterNum;
			this->weightVec[index_i] = trunc_weight(this->weightVec[index_i], 
				gravity *(this->sigma_w[index_i]));
		}
		Optimizer<FeatType, LabelType>::EndTrain();
	}

	template <typename FeatType, typename LabelType>
	void SSAROW<FeatType, LabelType>::SetParameterEx(float r) {
		this->r = r > 0 ? r : this->r;
	}

	//Change the dimension of weights
	template <typename FeatType, typename LabelType>
	void SSAROW<FeatType, LabelType>::UpdateWeightSize(IndexType newDim) {
		if (newDim < this->weightDim)
			return;
		else {
			this->sigma_w.reserve(newDim + 1);
			this->sigma_w.resize(newDim + 1);
			//set the rest to one
			this->sigma_w.set_value(this->sigma_w.begin + this->weightDim, 
				this->sigma_w.end,1);

			this->timeStamp.reserve(newDim + 1);
			this->timeStamp.resize(newDim + 1);
			//set the rest to zero
			this->timeStamp.zeros(this->timeStamp.begin + this->weightDim,
				this->timeStamp.end);

			Optimizer<FeatType,LabelType>::UpdateWeightSize(newDim);
		}
	}
	//get the best model parameter
	template <typename FeatType, typename LabelType>
	void SSAROW<FeatType, LabelType>::BestParameter() {
		//first learn the best learning rate
		float prevLambda = this->lambda;
		this->lambda = 0;

		//Select the best eta0
		float min_errorRate = 1;
		float bestr = 1;

		for (float r_temp = init_r_min;  r_temp <= init_r_max; r_temp *= init_r_step) {
			cout<<"r = "<<r_temp<<"\n";
			this->r = r_temp;
			float errorRate(0);
			errorRate = this->Train();

			if (errorRate < min_errorRate) {
				bestr = r_temp;
				min_errorRate = errorRate;
			}
			cout<<" mistake rate: "<<errorRate * 100<<" %\n";
		}

		this->r = bestr;
		this->lambda = prevLambda;
		cout<<"Best Parameter:\tr = "<<this->r<<"\n\n";
	}
}
