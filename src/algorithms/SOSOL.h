/************************************************************************
> File Name: SOSOL.h
> Copyright (C) 2013 Tim.WU<pcwu@ntu.edu.sg>
> Created Time: 18/01/2014 Monday 1:49:43 PM
> Functions: Sparse Online Learning
> Reference:
************************************************************************/

#pragma once


#include "../utils/util.h"
#include "Optimizer.h"
#include "../loss/SquaredHingeLoss.h"
#include <cmath>
#include <limits>

namespace SOL {
	template <typename FeatType, typename LabelType>
	class SOSOL: public Optimizer<FeatType, LabelType> {
        inline char Sign(float x) {
            if (x >= 0.f)
                return 1;
            else
                return -1;
        }
	protected:
		s_array<float> theta;
		s_array<float> A_inv_w;
		s_array<float> u_t; //TIM: can we use local variable for u_t?
		float r;
	protected:
		float (*pEta_time)(size_t t, float pt);

	public:
		SOSOL(DataSet<FeatType, LabelType> &dataset,
			LossFunction<FeatType, LabelType> &lossFunc);
		virtual ~SOSOL();

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
            printf("Learning Rate: %g\n", this->eta0);
            printf("Initial t  : %lu\n",this->initial_t);
            printf("Power t : %g\n",this->power_t);
			printf("lambda	: %g\n\n",this->lambda);
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
	SOSOL<FeatType, LabelType>::SOSOL(DataSet<FeatType, LabelType> &dataset,
		LossFunction<FeatType, LabelType> &lossFunc):
	Optimizer<FeatType, LabelType>(dataset, lossFunc) {
		this->id_str = "SOSOL";
		this->r = init_r;
		this->theta.resize(this->weightDim); // TIM: dim is d or d+1?
		this->A_inv_w.resize(this->weightDim);
		this->u_t.resize(this->weightDim);
	}

	template <typename FeatType, typename LabelType>
	SOSOL<FeatType, LabelType>::~SOSOL() {
//		if (this->lossFunc != NULL){
//			delete this->lossFunc;
//			this->lossFunc = NULL;
//		}
	}

	//this is the core of different updating algorithms
	//return the predict
	template <typename FeatType, typename LabelType>
	float SOSOL<FeatType,LabelType>::UpdateWeightVec(
		const DataPoint<FeatType, LabelType> &x) {

		    this->eta = this->eta0 / this->pEta_time(this->curIterNum, this->power_t);
		    size_t featDim = x.indexes.size();
		    IndexType index_i = 0;

			//obtain w_t
			for (size_t i = 0; i < featDim; i++) {
				index_i = x.indexes[i];
				A_inv_w[index_i] -= (A_inv_w[index_i] * x.features[i]);
				u_t[index_i] = A_inv_w[index_i] * this->theta[index_i];
				float abs_u_minus_lambda = fabsf(u_t[index_i]) - this->lambda;
				if (abs_u_minus_lambda < 0)
					abs_u_minus_lambda = 0;
				this->weightVec[index_i] = Sign(u_t[index_i]) * abs_u_minus_lambda;
			}

		    //predict
			float y = this->Predict(x);  // TIM: return is +1 or -1, right?
			//get gradient
//			float gt_i = this->lossFunc->GetGradient(x.label,y);
			float ell = this->lossFunc->GetLoss(x.label,y); //TIM: why other algorithms dont calculate loss in weights updating?

			//update
			if (ell > 0)
				for (size_t i = 0; i < featDim; i++) {
					this->theta[index_i] += this->eta * y * x.features[i];
				}

			return y;
	}

	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void SOSOL<FeatType, LabelType>::BeginTrain() {
		Optimizer<FeatType, LabelType>::BeginTrain();
		this->theta.zeros();
		this->A_inv_w.ones(); //TIM: correct?
		this->u_t.zeros();


		if (this->power_t == 0.5)
			this->pEta_time = pEta_sqrt;
		else if(this->power_t == 0)
			this->pEta_time = pEta_const;
		else if (this->power_t == 1)
			this->pEta_time = pEta_linear;
		else
			this->pEta_time = pEta_general;

		//what is the variable `timeStamp` for in other algorithms?
	}

	//called when a train ends
	template <typename FeatType, typename LabelType>
	void SOSOL<FeatType, LabelType>::EndTrain() {
		Optimizer<FeatType,LabelType>::EndTrain();
	}

	template <typename FeatType, typename LabelType>
	void SOSOL<FeatType, LabelType>::SetParameterEx(float r) {
		this->r = r > 0 ? r : this->r;
	}

	//Change the dimension of weights
	template <typename FeatType, typename LabelType>
	void SOSOL<FeatType, LabelType>::UpdateWeightSize(IndexType newDim) {
		if (newDim < this->weightDim)
			return;
		else {
			this->theta.reserve(newDim + 1);
			this->theta.resize(newDim + 1);
			//set the rest to zero
			this->theta.zeros(this->theta.begin + this->weightDim,
							this->theta.end);

			this->A_inv_w.reserve(newDim + 1);
			this->A_inv_w.resize(newDim + 1);
			//set the rest to one
			this->A_inv_w.set_value(this->A_inv_w.begin + this->weightDim,
				this->A_inv_w.end,1);

			this->u_t.reserve(newDim + 1);
			this->u_t.resize(newDim + 1);
			//set the rest to zero
			this->u_t.zeros(this->u_t.begin + this->weightDim,
				this->u_t.end);

			Optimizer<FeatType,LabelType>::UpdateWeightSize(newDim);
		}
	}
	//get the best model parameter
	template <typename FeatType, typename LabelType>
	void SOSOL<FeatType, LabelType>::BestParameter() {
		Optimizer<FeatType,LabelType>::BestParameter();
	}
}
