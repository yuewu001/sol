/*************************************************************************
> File Name: SCW_RDA.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/11/27 17:10:06
> Functions: Exact Soft Confidence-Weighted Learning
************************************************************************/
#ifndef HEADER_SCW_RDA
#define HEADER_SCW_RDA

#include "Optimizer.h"
#include <cmath>

namespace SOL {
	template <typename FeatType, typename LabelType>
	class SCW_RDA: public Optimizer<FeatType, LabelType> {
	protected:
		s_array<float> sigma_w;
		s_array<float> u_t;
		s_array<float> gravity_t;

		float C;

		float phi;
		float phi_sq;
		float phi_44;
		float psi;
		float zeta;

	public:
		SCW_RDA(DataSet<FeatType, LabelType> &dataset, 
			LossFunction<FeatType, LabelType> &lossFunc);
		~SCW_RDA();

	public:
		//set parameters for specific optimizers
		void SetParameterEx(float phi = -1,float r = -1);

	protected:
		//this is the core of different updating algorithms
		virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);

		//Change the dimension of weights
		virtual void UpdateWeightSize(IndexType newDim);

		//reset
		virtual void BeginTrain();
		//called when a train ends
		virtual void EndTrain();

		//try and get the best parameter
		virtual void BestParameter(){}

	protected:
		//update phi: related probability of correct prediction
		void update_phi(float new_val){
			this->phi = new_val;
			this->phi_sq = this->phi * this->phi;
			this->phi_44 = this->phi_sq * this->phi_sq / 4.f;
			this->zeta = 1 + this->phi_sq;
			this->psi = 1 + this->phi_sq / 2.f;
		}
	};

	template <typename FeatType, typename LabelType>
	SCW_RDA<FeatType, LabelType>::SCW_RDA(DataSet<FeatType, LabelType> &dataset, 
		LossFunction<FeatType, LabelType> &lossFunc):
	Optimizer<FeatType, LabelType>(dataset, lossFunc){
		this->id_str = "Soft Confidence Weighted";
		this->sigma_w.resize(this->weightDim);
		this->u_t.resize(this->weightDim);
		this->gravity_t.resize(this->weightDim);

		this->C = 0.5f / init_r;
		this->update_phi(init_phi);
	}

	template <typename FeatType, typename LabelType>
	SCW_RDA<FeatType, LabelType>::~SCW_RDA() {
	}

	//this is the core of different updating algorithms
	template <typename FeatType, typename LabelType>
	float SCW_RDA<FeatType,LabelType>::UpdateWeightVec(
		const DataPoint<FeatType, LabelType> &x) {
			size_t featDim = x.indexes.size();
			IndexType index_i = 0;

			//obtain w_t
			for (size_t i = 0; i < featDim; i++) {
				index_i = x.indexes[i];
				//lazy update
				this->weightVec[index_i] = -this->sigma_w[index_i] *
					trunc_weight(u_t[index_i], this->gravity_t[index_i] * (this->curIterNum - 1));
			}

			float vt = 0;
			for (size_t i = 0; i < featDim; i++){
				vt += x.features[i] * x.features[i] * this->sigma_w[x.indexes[i]];
			}
			//predict 
			float y = this->Predict(x);
			float mt = x.label * y;
			float alpha_t = (-mt * this->psi + sqrtf(mt * mt * this->phi_44 + vt * this->phi_sq * this->zeta));
			alpha_t  /= (vt * this->zeta);
			alpha_t = (std::max)(0.f, alpha_t);
			if (alpha_t > 0){
				alpha_t = (std::min)(C, alpha_t);
				float ut = -alpha_t * vt * this->phi + sqrtf(alpha_t * alpha_t * vt * vt * this->phi_sq + 4.f * vt);
				ut = ut * ut / 4.f;
				float beta_t = alpha_t * this->phi / (sqrtf(ut) + vt * alpha_t * this->phi);
				float gravity = this->lambda * alpha_t;
				float gt = -x.label * alpha_t;
				//update
				for (size_t i = 0; i < featDim; i++) {
					index_i = x.indexes[i];
					//this->weightVec[index_i] += alpha_t * x.label * x.features[i] * this->sigma_w[index_i];
					this->u_t[index_i] += gt * x.features[i];
					this->gravity_t[index_i] = gravity;
					//update sigma_w
					this->sigma_w[index_i] -= beta_t * this->sigma_w[index_i] * this->sigma_w[index_i] 
					* x.features[i] * x.features[i];
				}
				//bias term
				this->u_t[index_i] += gt;
				//this->weightVec[0] +=  alpha_t * x.label * this->sigma_w[0];
				this->weightVec[0] = -u_t[0] * this->sigma_w[0]; 
				this->sigma_w[0] -= beta_t * this->sigma_w[0] * this->sigma_w[0]; 
			}
			return y;
	}

	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void SCW_RDA<FeatType, LabelType>::BeginTrain() {
		Optimizer<FeatType, LabelType>::BeginTrain();
		this->u_t.zeros();
		this->gravity_t.zeros();
		this->sigma_w.set_value(1);
	}

	//called when a train ends
	template <typename FeatType, typename LabelType>
	void SCW_RDA<FeatType, LabelType>::EndTrain() {
		Optimizer<FeatType,LabelType>::EndTrain();
	}

	//set parameters for specific optimizers
	template <typename FeatType, typename LabelType>
	void SCW_RDA<FeatType, LabelType>::SetParameterEx(float phi,float r) {
		if (phi > 0)
			this->update_phi(phi);
		this->C = r > 0 ? 0.5f / r : this->C;
	}

	//Change the dimension of weights
	template <typename FeatType, typename LabelType>
	void SCW_RDA<FeatType, LabelType>::UpdateWeightSize(IndexType newDim) {
		if (newDim < this->weightDim)
			return;
		else {
			this->sigma_w.reserve(newDim + 1);
			this->sigma_w.resize(newDim + 1);
			//set the rest to one
			this->sigma_w.set_value(this->sigma_w.begin + this->weightDim, 
				this->sigma_w.end,1);

			this->u_t.reserve(newDim + 1);
			this->u_t.resize(newDim + 1);
			//set the rest to zero
			this->u_t.zeros(this->u_t.begin + this->weightDim,
				this->u_t.end);

			this->gravity_t.reserve(newDim + 1);
			this->gravity_t.resize(newDim + 1);
			//set the rest to zero
			this->gravity_t.zeros(this->gravity_t.begin + this->weightDim,
				this->gravity_t.end);

			Optimizer<FeatType,LabelType>::UpdateWeightSize(newDim);
		}
	}
}

#endif
