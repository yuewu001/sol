/*************************************************************************
> File Name: Sparse Diagonal AROW for adaptive feature selection
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2014/2/11 Tuesday 17:52:54
> Functions: Diagonal Adaptive Regularization of Weight Vectors for adaptive feature selection
> Reference:
Crammer, Koby, Alex Kulesza, and Mark Dredze. "Adaptive regularization
of weight vectors." Machine Learning (2009): 1-33.
************************************************************************/

#ifndef HEADER_AROW_ADA_FS
#define HEADER_AROW_ADA_FS


#include "../utils/util.h"
#include "Optimizer.h"
#include "MaxHeap.h"
#include <algorithm>
#include <math.h>
#include <vector>

namespace SOL {
	template <typename FeatType, typename LabelType>
	class AROW_AdaFS : public Optimizer<FeatType, LabelType> {
	protected:
		float r;
		s_array<float> sigma_w;
		s_array<float> sigma_w_full;

		s_array<float> w_full;

		MaxHeap<float> heap;

		IndexType K; //keep top K elemetns
		size_t change_step; //change the update step

		size_t errNum;
		size_t errNum_full;

		size_t delta_update_num;

		float tolerance;
	public:
		AROW_AdaFS(DataSet<FeatType, LabelType> &dataset,
			LossFunction<FeatType, LabelType> &lossFunc);
		virtual ~AROW_AdaFS();

	public:
		void SetParameterEx(int k, float r = -1);
		//select the best parameters for the model
		virtual void BestParameter();
		/**
		* PrintOptInfo print the info of optimization algorithm
		*/
		virtual void PrintOptInfo() const {
			printf("--------------------------------------------------\n");
			printf("Algorithm: %s\n\n", this->Id_Str().c_str());
			printf("k:\t%d\n", this->K);
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
		int Update_full(const DataPoint<FeatType, LabelType> &x);
	};

	template <typename FeatType, typename LabelType>
	AROW_AdaFS<FeatType, LabelType>::AROW_AdaFS(DataSet<FeatType, LabelType> &dataset,
		LossFunction<FeatType, LabelType> &lossFunc) :
		Optimizer<FeatType, LabelType>(dataset, lossFunc){
			this->id_str = "AROW for Feature Selection";
			this->r = init_r;
			this->K = 0;
			this->w_full.resize(this->weightDim);
			this->tolerance = 1.05f;
		}

	template <typename FeatType, typename LabelType>
	AROW_AdaFS<FeatType, LabelType>::~AROW_AdaFS() {
	}


	//this is the core of different updating algorithms
	//return the predict
	template <typename FeatType, typename LabelType>
	float AROW_AdaFS<FeatType, LabelType>::UpdateWeightVec(
		const DataPoint<FeatType, LabelType> &x) {
		this->errNum_full += this->Update_full(x);

		float y = this->Predict(x);
		//y /= this->curIterNum;
		float alpha_t = 1 - x.label * y;
		if (alpha_t > 0){
			IndexType index_i = 0;
			size_t featDim = x.indexes.size();
			//calculate beta_t
			float beta_t = this->r;
			for (size_t i = 0; i < featDim; i++){
				beta_t += x.features[i] * x.features[i] * this->sigma_w[x.indexes[i]];
			}
			beta_t = 1.f / beta_t;
			alpha_t *= beta_t;
			for (size_t i = 0; i < featDim; i++){
				index_i = x.indexes[i];
				if (this->heap.is_topK(index_i - 1)){
					//update u_t
					this->weightVec[index_i] += alpha_t *
						this->sigma_w[index_i] * x.label * x.features[i];
				}
				else{
					this->weightVec[index_i] = 0;
				}
				//update sigma_w
				//this->sigma_w[index_i] -= beta_t * this->sigma_w[index_i] * this->sigma_w[index_i] * x.features[i] * x.features[i];
				this->sigma_w[index_i] *= this->r / (this->r +
					this->sigma_w[index_i] * x.features[i] * x.features[i]);
				IndexType ret_id;
				this->heap.UpdateHeap(index_i - 1, ret_id);
				//heap.Output(); 
			}
			//bias term
			this->weightVec[0] += alpha_t * this->sigma_w[0] * x.label;
			//this->sigma_w[0] -= beta_t * this->sigma_w[0] * this->sigma_w[0];
			this->sigma_w[0] *= this->r / (this->r + this->sigma_w[0]);

			this->errNum++;
		}
		this->delta_update_num++;
		//if (this->delta_update_num > this->change_step){
		if (this->delta_update_num > 20000){
			IndexType newsize = 0;
			if (this->errNum < this->errNum_full * this->tolerance){
				newsize = this->heap.GetK() / 2;
				this->change_step = newsize;
				newsize = newsize > 0 ? newsize : 1;
			}
			else{
				newsize = (this->heap.GetK() + this->weightDim - 1) / 2;
				this->change_step = this->weightDim - 1 - newsize;
			}
			//copy sigma and w
			memcpy(this->weightVec.begin, this->w_full.begin, sizeof(float)* this->w_full.size());
			memcpy(this->sigma_w.begin, this->sigma_w_full.begin, sizeof(float)* this->sigma_w_full.size());
			this->heap.ResizeHeap(newsize);
			this->errNum = this->errNum_full = 0;
			this->delta_update_num = 0;
		}

		return y;
	}

	template <typename FeatType, typename LabelType>
	int AROW_AdaFS<FeatType, LabelType>::Update_full(const DataPoint<FeatType, LabelType> &x){
		float y = this->w_full[0];
		size_t featDim = x.indexes.size();
		for (size_t i = 0; i < featDim; i++){
			y += this->w_full[x.indexes[i]] * x.features[i];
		}

		float alpha_t = 1 - x.label * y;

		IndexType index_i = 0;
		if (alpha_t > 0){
			//calculate beta_t
			float beta_t = this->r;
			for (size_t i = 0; i < featDim; i++)
				beta_t += x.features[i] * x.features[i] * this->sigma_w_full[x.indexes[i]];
			beta_t = 1.f / beta_t;
			alpha_t *= beta_t;
			for (size_t i = 0; i < featDim; i++){
				index_i = x.indexes[i];
				this->w_full[index_i] += alpha_t * this->sigma_w_full[index_i] * x.label * x.features[i];
				//update sigma_w
				this->sigma_w_full[index_i] *= this->r / (this->r +
					this->sigma_w_full[index_i] * x.features[i] * x.features[i]);
			}
			//bias term
			this->w_full[0] += alpha_t * this->sigma_w_full[0] * x.label;
			this->sigma_w_full[0] *= this->r / (this->r + this->sigma_w_full[0]);
			return 1;
		}
		return 0;
	}

	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void AROW_AdaFS<FeatType, LabelType>::BeginTrain() {
		Optimizer<FeatType, LabelType>::BeginTrain();
		if (this->K < 1){
			cerr << "Please specify a valid number of weights to keep!\n";
			cerr << "current number: " << this->K << endl;
			exit(0);
		}
		if (this->weightDim < 2 * this->K + 1){
			this->UpdateWeightSize(2 * this->K); //remove the bais term
		}
		this->sigma_w.set_value(1);
		this->sigma_w_full.set_value(1);
		heap.Init(this->weightDim - 1, this->K, this->sigma_w.begin + 1);
		this->w_full.zeros();
		this->errNum = 0;
		this->errNum_full = 0;
		this->delta_update_num = 0;
		this->change_step = this->K;
		//this->is_normalize = true; //AROW requires normalization
	}

	//called when a train ends
	template <typename FeatType, typename LabelType>
	void AROW_AdaFS<FeatType, LabelType>::EndTrain() {
		Optimizer<FeatType, LabelType>::EndTrain();
		cout << "learned buffer size: " << this->heap.GetK() << endl;
	}

	template <typename FeatType, typename LabelType>
	void AROW_AdaFS<FeatType, LabelType>::SetParameterEx(int k, float r) {
		if (k < 1){
			cerr << "Please specify a valid number of weights to keep!\n";
			cerr << "current number: " << this->K << endl;
			exit(0);
		}
		else
			this->K = k;
		this->r = r > 0 ? r : this->r;
	}

	//Change the dimension of weights
	template <typename FeatType, typename LabelType>
	void AROW_AdaFS<FeatType, LabelType>::UpdateWeightSize(IndexType newDim) {
		if (newDim < this->weightDim)
			return;
		else {
			this->sigma_w.reserve(newDim + 1);
			this->sigma_w.resize(newDim + 1);  //reserve the 0-th
			//set the rest to 1
			this->sigma_w.set_value(this->sigma_w.begin + this->weightDim,
				this->sigma_w.end, 1);

			this->sigma_w_full.reserve(newDim + 1);
			this->sigma_w_full.resize(newDim + 1);  //reserve the 0-th
			//set the rest to 1
			this->sigma_w_full.set_value(this->sigma_w_full.begin + this->weightDim,
				this->sigma_w_full.end, 1);

			heap.UpdateDataNum(newDim, this->sigma_w.begin + 1);

			this->w_full.reserve(newDim + 1);
			this->w_full.resize(newDim + 1);
			//set the new value to zero
			this->w_full.zeros(this->w_full.begin + this->weightDim,
				this->w_full.end);

			Optimizer<FeatType, LabelType>::UpdateWeightSize(newDim);
		}
	}

	//get the best model parameter
	template <typename FeatType, typename LabelType>
	void AROW_AdaFS<FeatType, LabelType>::BestParameter() {
		//first learn the best learning rate
		float prevLambda = this->lambda;
		this->lambda = 0;

		//Select the best eta0
		float min_errorRate = 1;
		float bestr = 1;

		for (float r_temp = init_r_min; r_temp <= init_r_max; r_temp *= init_r_step) {
			cout << "r = " << r_temp << "\n";
			this->r = r_temp;
			float errorRate(0);
			errorRate = this->Train();

			if (errorRate < min_errorRate) {
				bestr = r_temp;
				min_errorRate = errorRate;
			}
			cout << " mistake rate: " << errorRate * 100 << " %\n";
		}

		this->r = bestr;
		this->lambda = prevLambda;
		cout << "Best Parameter:\tr = " << this->r << "\n\n";
	}
}
#endif
