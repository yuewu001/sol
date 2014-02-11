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
#include "HeapList.h"
#include <algorithm>
#include <math.h>
#include <vector>

namespace SOL {
	template <typename FeatType, typename LabelType>
	class AROW_AdaFS : public Optimizer<FeatType, LabelType> {
	protected:
		float r;
		s_array<float> sigma_w;
		s_array<float> w_small;
		s_array<float> w_large;

		HeapList<float> heap_small;
		HeapList<float> heap;
		HeapList<float> heap_large;

		IndexType K; //keep top K elemetns
		size_t heap_gap_small; //length gap between heap_small and heap
		size_t heap_gap_large; //length gap between heap and heap_large

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
		float Update(const DataPoint<FeatType, LabelType> &x, s_array<float> &w, HeapList<float>& h, float &beta_t);
	};

	template <typename FeatType, typename LabelType>
	AROW_AdaFS<FeatType, LabelType>::AROW_AdaFS(DataSet<FeatType, LabelType> &dataset,
		LossFunction<FeatType, LabelType> &lossFunc) :
		Optimizer<FeatType, LabelType>(dataset, lossFunc){
			this->id_str = "AROW for Feature Selection";
			this->r = init_r;
			this->K = 0;
			this->sigma_w.resize(this->weightDim);
		}

	template <typename FeatType, typename LabelType>
	AROW_AdaFS<FeatType, LabelType>::~AROW_AdaFS() {
	}


	//this is the core of different updating algorithms
	//return the predict
	template <typename FeatType, typename LabelType>
	float AROW_AdaFS<FeatType, LabelType>::UpdateWeightVec(
		const DataPoint<FeatType, LabelType> &x) {
		float y = this->Predict(x);
		//y /= this->curIterNum;
		float alpha_t = 1 - x.label * y;
		float beta_t = 0;
		if (alpha_t > 0){
			IndexType index_i = 0;
			size_t featDim = x.indexes.size();
			//calculate beta_t
			beta_t = this->r;
			for (size_t i = 0; i < featDim; i++){
				beta_t += x.features[i] * x.features[i] * this->sigma_w[x.indexes[i]];
			}
			beta_t = 1.f / beta_t;
			alpha_t *= beta_t;
			for (size_t i = 0; i < featDim; i++){
				index_i = x.indexes[i];
				if (this->heap.is_topK(index_i - 1)){
					//update u_t
					this->weightVec[index_i] += alpha_t * this->sigma_w[index_i] * x.label * x.features[i];
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
		}
		return y;
	}

	template <typename FeatType, typename LabelType>
	float AROW_AdaFS<FeatType, LabelType>::Update(const DataPoint<FeatType, LabelType> &x,
		s_array<float> &w, HeapList<float>& h, float &beta_t){
		float y = w[0];
		size_t featDim = x.indexes.size();
		for (size_t i = 0; i < featDim; i++)
			y += w[x.indexes[i]] * x.features[i];

		float alpha_t = 1 - x.label * y;

		IndexType index_i = 0;
		if (alpha_t > 0){
			//calculate beta_t
				float beta_t = this->r;
				for (size_t i = 0; i < featDim; i++){
					beta_t += x.features[i] * x.features[i] * this->s[x.indexes[i]];
				}
				beta_t = 1.f / beta_t;
			}
			alpha_t *= beta_t;
			for (size_t i = 0; i < featDim; i++){
				index_i = x.indexes[i];
				if (h.is_topK(index_i - 1)){
					//update u_t
					this->w[index_i] += alpha_t * this->s[index_i] * x.label * x.features[i];
				}
				else{
					this->w[index_i] = 0;
				}
				//update sigma_w
				//this->sigma_w[index_i] -= beta_t * this->sigma_w[index_i] * this->sigma_w[index_i] * x.features[i] * x.features[i];
				this->s[index_i] *= this->r / (this->r +
					this->s[index_i] * x.features[i] * x.features[i]);
				IndexType ret_id;
				this->h.UpdateHeap(index_i - 1, ret_id);
			}
			//bias term
			this->w[0] += alpha_t * this->s[0] * x.label;
			this->s[0] *= this->r / (this->r + this->s[0]);
		}
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
		if (this->weightDim < this->K + 1){
			this->UpdateWeightSize(this->K); //remove the bais term
		}
		this->sigma_w.set_value(1);
		heap.Init(this->weightDim - 1, this->K, this->sigma_w.begin + 1);
		heap_small.Init(this->weightDim - 1, this->K, this->sigma_w.begin + 1);
		heap_large.Init(this->weightDim - 1, this->K, this->sigma_w.begin + 1);
		this->heap_gap_large = 0;
		this->heap_gap_small = 0;
		this->w_large.zeros();
		this->w_small.zeros();
		//this->is_normalize = true; //AROW requires normalization
	}

	//called when a train ends
	template <typename FeatType, typename LabelType>
	void AROW_AdaFS<FeatType, LabelType>::EndTrain() {
		Optimizer<FeatType, LabelType>::EndTrain();
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

			heap.UpdateDataNum(newDim, this->sigma_w.begin + 1);
			heap_small.UpdateDataNum(newDim, this->sigma_w.begin + 1);
			heap_large.UpdateDataNum(newDim, this->sigma_w.begin + 1);

			this->w_small.reserve(newDim + 1);
			this->w_small.resize(newDim + 1);
			//set the new value to zero
			this->w_small.zeros(this->w_small.begin + this->weightDim,
				this->w_small.end);

			this->w_large.reserve(newDim + 1);
			this->w_large.resize(newDim + 1);
			//set the new value to zero
			this->w_large.zeros(this->w_large.begin + this->weightDim,
				this->w_large.end);

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
