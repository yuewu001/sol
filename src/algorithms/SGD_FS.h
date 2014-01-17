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
			printf("--------------------------------------------------\n");
			printf("Algorithm: %s\n\n",this->Id_Str().c_str());
			printf("eta:\t%.2f\n", this->eta0);
			printf("Power t : %g\n",this->power_t);
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
		void QuickSort(float *a, int low, int high, IndexType *m_index);
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

			//truncate
			//sort
			for (size_t i = 0; i < this->weightDim; i++){
				this->abs_weightVec[index_i] = fabsf(this->weightVec[this->index_vec[index_i]]);
			}
			this->QuickSort(this->abs_weightVec.begin,1,
				this->weightDim - 1,this->index_vec.begin); 
			//truncate
			for (int i = this->K + 1; i < this->weightDim; i++){
				this->weightVec[this->index_vec[i]] = 0;
			}
			return y;
	}

	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void SGD_FS<FeatType, LabelType>::BeginTrain() {
		Optimizer<FeatType, LabelType>::BeginTrain();
		if (this->K < 1){
			cerr<<"Please specify a valid number of weights to keep!\n";
			cerr<<"current number: "<<this->K<<endl;
			exit(0);
		}
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
		if (k < 1){
			cerr<<"Please specify a valid number of weights to keep!\n";
			cerr<<"current number: "<<this->K<<endl;
			exit(0);
		}
		else
			this->K = k;
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
	void SGD_FS<FeatType, LabelType>::QuickSort(float *a, int low, int high, IndexType *m_index) {// from great to small
		int i = low;
		int j = high;
		float temp = a[low]; 
		int temp_ind = m_index[low];

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

		if (low < i) 
			QuickSort(a, low, i-1, m_index);  
		if (i < high) 
			QuickSort(a, j+1, high, m_index);  
	}
}

#endif
