/*************************************************************************
  > File Name: SGD_FS.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Tue 17 Dec 2013 04:59:06 AM EST
  > Descriptions: Online Feature Selection with PE_trunc
  > Reference: Online Feature Selection and its applications
 ************************************************************************/
#ifndef HEADER_OPTIMISER_SGD_FS
#define HEADER_OPTIMISER_SGD_FS

#include "../utils/util.h"
#include "Optimizer.h"
#include <algorithm>
#include <math.h>
#include <queue>

namespace SOL {
	template <typename FeatType, typename LabelType>
	class SGD_FS: public Optimizer<FeatType, LabelType> {
	protected:
		IndexType K; //keep top K elemetns

		s_array<IndexType> index_vec;
		s_array<float> abs_weightVec;

		float (*pEta_time)(size_t t, float pt);
	public:
		SGD_FS(DataSet<FeatType, LabelType> &dataset, 
			LossFunction<FeatType, LabelType> &lossFunc);
		virtual ~SGD_FS();

	public:
		void SetParameterEx(int k);
		/**
		* PrintOptInfo print the info of optimization algorithm
		*/
		virtual void PrintOptInfo() const {
			Optimizer<FeatType,LabelType>::PrintOptInfo();
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
	SGD_FS<FeatType, LabelType>::SGD_FS(DataSet<FeatType, LabelType> &dataset, 
		LossFunction<FeatType, LabelType> &lossFunc):
	Optimizer<FeatType, LabelType>(dataset, lossFunc){
		this->id_str = "SGD_FS";
		this->K = 0;

		this->index_vec.resize(this->weightDim);
		this->index_vec.zeros();
		this->abs_weightVec.resize(this->weightDim);
	}

	template <typename FeatType, typename LabelType>
	SGD_FS<FeatType, LabelType>::~SGD_FS() {
	}

	//this is the core of different updating algorithms
	//return the predict
	template <typename FeatType, typename LabelType>
	float SGD_FS<FeatType,LabelType>::UpdateWeightVec(
		const DataPoint<FeatType, LabelType> &x) {
			//we use the oposite of w
			float y = this->Predict(x);
			size_t featDim = x.indexes.size();
			this->eta = this->eta0 / this->pEta_time(this->curIterNum, this->power_t);

			float gt_i = this->lossFunc->GetGradient(x.label,y);
			if (gt_i == 0){
				return y;
			}

			IndexType index_i = 0;
			//update with sgd
			for (size_t i = 0; i < featDim; i++) {
                index_i = x.indexes[i];
				this->weightVec[index_i] -= this->eta * gt_i * x.features[i];
			}
			//update bias 
			this->weightVec[0] -= this->eta * gt_i;
			if (this->K > 0){
				//truncate
				//sort
				for (size_t i = 0; i < this->weightDim; i++){
					this->abs_weightVec[index_i] = fabsf(this->weightVec[this->index_vec[index_i]]);
				}
				this->Sort(this->abs_weightVec.begin,1,
					this->weightDim - 1,this->index_vec.begin); 
				//truncate
				for (IndexType i = this->K + 1; i < this->weightDim; i++){
					this->weightVec[this->index_vec[i]] = 0;
				}
			}
			return y;
	}

	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void SGD_FS<FeatType, LabelType>::BeginTrain() {
		Optimizer<FeatType, LabelType>::BeginTrain();
		if (this->power_t == 0.5)
			this->pEta_time = pEta_sqrt;
		else if(this->power_t == 0)
			this->pEta_time = pEta_const;
		else if (this->power_t == 1)
			this->pEta_time = pEta_linear;
		else
			this->pEta_time = pEta_general;
	}

	//called when a train ends
	template <typename FeatType, typename LabelType>
	void SGD_FS<FeatType, LabelType>::EndTrain() {
		Optimizer<FeatType, LabelType>::EndTrain();
	}

	template <typename FeatType, typename LabelType>
	void SGD_FS<FeatType, LabelType>::SetParameterEx(int k) {
		this->K = k > 0 ? k : this->K;
	}

	//Change the dimension of weights
	template <typename FeatType, typename LabelType>
	void SGD_FS<FeatType, LabelType>::UpdateWeightSize(IndexType newDim) {
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
	void SGD_FS<FeatType, LabelType>::Sort(float *a, IndexType low, IndexType high, IndexType *m_index) {// from great to small
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
	void SGD_FS<FeatType, LabelType>::QuickSort(float *a, IndexType low, IndexType high, IndexType *m_index) {// from great to small
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
