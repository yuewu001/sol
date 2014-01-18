/*************************************************************************
  > File Name: OFSGD.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Tue 17 Dec 2013 04:59:06 AM EST
  > Descriptions: Online Feature Selection
  > Reference: Online Feature Selection and its applications
 ************************************************************************/
#ifndef HEADER_OPTIMISER_OFSGD
#define HEADER_OPTIMISER_OFSGD

#include "../utils/util.h"
#include "Optimizer.h"
#include <algorithm>
#include <math.h>
#include <queue>

namespace SOL {
	template <typename FeatType, typename LabelType>
	class OFSGD: public Optimizer<FeatType, LabelType> {
	protected:
		IndexType K; //keep top K elemetns

		float w_norm;
		float norm_coeff;

		float shrinkage;
		float delta;
		s_array<IndexType> index_vec;
		s_array<float> abs_weightVec;

	public:
		OFSGD(DataSet<FeatType, LabelType> &dataset, 
			LossFunction<FeatType, LabelType> &lossFunc);
		virtual ~OFSGD();

	public:
		void SetParameterEx(int k, float delta);
		/**
		* PrintOptInfo print the info of optimization algorithm
		*/
		virtual void PrintOptInfo() const {
			printf("--------------------------------------------------\n");
			printf("Algorithm: %s\n\n",this->Id_Str().c_str());
			printf("eta:\t%.2f\n", this->eta0);
			printf("delta:\t%.2f\n", this->delta);
			printf("K:\t%d\n\n", this->K);
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

	protected:
		void QuickSort(float *a, IndexType low, IndexType high, IndexType *m_index);
		void Sort(float *a, IndexType low, IndexType high, IndexType *m_index);
	};

	template <typename FeatType, typename LabelType>
	OFSGD<FeatType, LabelType>::OFSGD(DataSet<FeatType, LabelType> &dataset, 
		LossFunction<FeatType, LabelType> &lossFunc):
	Optimizer<FeatType, LabelType>(dataset, lossFunc){
		this->id_str = "OFSGD";
		this->K = 0;
		this->delta = init_ofs_delta;

		this->index_vec.resize(this->weightDim);
		this->index_vec.zeros();
		this->abs_weightVec.resize(this->weightDim);
	}

	template <typename FeatType, typename LabelType>
	OFSGD<FeatType, LabelType>::~OFSGD() {
	}

	//this is the core of different updating algorithms
	//return the predict
	template <typename FeatType, typename LabelType>
	float OFSGD<FeatType,LabelType>::UpdateWeightVec(
		const DataPoint<FeatType, LabelType> &x) {
			//we use the oposite of w
			float y = this->Predict(x);
			size_t featDim = x.indexes.size();
			this->eta = this->eta0;

			//shrinkage
			for (size_t i = 0; i < this->weightDim; i++){
				this->weightVec[this->index_vec[i]] *= shrinkage;
			}

			float gt_i = this->lossFunc->GetGradient(x.label,y);
			if (gt_i == 0){
				return y;
			}

			IndexType index_i = 0;
			
			//update with sgd
			for (size_t i = 0; i < featDim; i++) {
				this->weightVec[x.indexes[i]] -= this->eta * gt_i * x.features[i];
			}
			//update bias 
			this->weightVec[0] -= this->eta * gt_i;

			w_norm = 0;
			for (size_t i = 0; i < this->weightDim; i++)
				w_norm += this->weightVec[i] * this->weightVec[i];
			w_norm = sqrtf(w_norm);
			//shrinkage
			float coeff = this->norm_coeff / sqrtf(w_norm);
			if (coeff < 1){
				for (IndexType i = 0; i < this->weightDim; i++)
					this->weightVec[i] *= coeff;
			}
			//truncate
			//sort
			for (size_t i = 0; i < this->weightDim; i++){
				this->abs_weightVec[i] = fabsf(this->weightVec[this->index_vec[i]]);
			}
			this->Sort(this->abs_weightVec.begin,1,
				this->weightDim - 1,this->index_vec.begin); 
			//truncate
			for (IndexType i = this->K + 1; i < this->weightDim; i++){
				this->weightVec[this->index_vec[i]] = 0;
			}
			return y;
	}

	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void OFSGD<FeatType, LabelType>::BeginTrain() {
		Optimizer<FeatType, LabelType>::BeginTrain();
		if (this->K < 1){
			cerr<<"Please specify a valid number of weights to keep!\n";
			cerr<<"current number: "<<this->K<<endl;
			exit(0);
		}
		this->w_norm = 0;
		this->norm_coeff = 1.f / sqrtf(this->delta);

		this->shrinkage = 1.f - this->delta * this->eta0;
		this->power_t = 0;
	}

	//called when a train ends
	template <typename FeatType, typename LabelType>
	void OFSGD<FeatType, LabelType>::EndTrain() {
		Optimizer<FeatType, LabelType>::EndTrain();
	}

	template <typename FeatType, typename LabelType>
	void OFSGD<FeatType, LabelType>::SetParameterEx(int k, float Delta) {
		if (k < 1){
			cerr<<"Please specify a valid number of weights to keep!\n";
			cerr<<"current number: "<<this->K<<endl;
			exit(0);
		}
		else
			this->K = k;
		this->delta = Delta > 0 ? Delta : this->delta;
	}

	//Change the dimension of weights
	template <typename FeatType, typename LabelType>
	void OFSGD<FeatType, LabelType>::UpdateWeightSize(IndexType newDim) {
		if (newDim < this->weightDim)
			return;
		else {
			this->index_vec.reserve(newDim + 1);
			this->index_vec.resize(newDim + 1);
			for (size_t index_i = this->weightDim; index_i < newDim + 1; index_i++)
				this->index_vec[index_i] = index_i;

			this->abs_weightVec.reserve(newDim + 1);
			this->abs_weightVec.resize(newDim + 1);

			Optimizer<FeatType,LabelType>::UpdateWeightSize(newDim);
		}
	}

	template <typename FeatType, typename LabelType>
	void OFSGD<FeatType, LabelType>::Sort(float *a, IndexType low, IndexType high, IndexType *m_index) {// from great to small
		//move all zeros to the end
		IndexType i = low;
		IndexType j = high;
		float temp = a[low]; 
		IndexType temp_ind = m_index[low];

		//move all zeros to the end
		while(i < j){
			while(i < j && a[i] != 0) i++;
			while(i < j && a[j] == 0) j--;
			if (i < j){ //swap a[i], a[j]
				temp = a[i]; temp_ind = m_index[i];
				a[i] = a[j]; m_index[i] = m_index[j];
				a[j] = temp; m_index[j] = temp_ind;
			}
		}
		if (j > this->K){
			this->QuickSort(a,low,j,m_index);
		}
	}

	template <typename FeatType, typename LabelType>
	void OFSGD<FeatType, LabelType>::QuickSort(float *a, IndexType low, IndexType high, IndexType *m_index) {// from great to small
		//one pass maopao
		float temp ;
		IndexType temp_ind; 
		IndexType i = low;
		IndexType j = high;
		//one pass maopao
		bool is_sorted = true;
		for (i = low; i < high; i++){
			if (a[i] < a[i+1]){ //swap
				temp = a[i];
				temp_ind = m_index[i];
				is_sorted = false;
				break;
			}
		}
		if (is_sorted == true)
			return;

		a[i] = a[low];
		m_index[i] = m_index[low];
		a[low] = temp;
		m_index[low] = temp_ind;

		i = low;

		while (i < j) {
			while ((i < j) && (temp >= a[j]))
				j--;
			if(i<j) {
				a[i] = a[j];
				m_index[i] = m_index[j];
				i ++;
			}

			while (i<j && (a[i] >= temp)) 
				i++;
			if (i<j) {
				a[j] = a[i];
				m_index[j] = m_index[i];

				j--;
			}
		}

		a[i] = temp;
		m_index[i] = temp_ind;

		if (i > this->K && low < i) {
			QuickSort(a, low, i-1, m_index);  
		}
		else if (i < this->K && i < high) {
			QuickSort(a, j+1, high, m_index);  
		}
	}
}

#endif
