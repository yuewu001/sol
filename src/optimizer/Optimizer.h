/*************************************************************************
> File Name: Optimizer.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/18 星期日 16:04:18
> Functions: Base class for different algorithms to do online learning
************************************************************************/

#pragma once
#include "../data/DataSet.h"
#include "../loss/LossFunction.h"
#include "../common/init_param.h"

#include <stdio.h>
#include <math.h>


/**
*  namespace: Sparse Online Learning
*/
namespace SOL {
	template <typename FeatType, typename LabelType> class Optimizer {
		//Iteration 
	protected:
		//iteration number
		size_t curIterNum;
        size_t initial_t;
        float power_t;
		//parameters
		float lambda;
		float eta0; //learning rate
        float eta;

		DataSet<FeatType, LabelType> &dataSet;

		bool is_normalize;

		//weight vector
	protected:
		//the first element is zero
		s_array<float> weightVec;
		//weight dimenstion: can be the same to feature, or with an extra bias
		IndexType weightDim;

		//For sparse
	protected:
		float sparse_soft_thresh;

	protected:
		LossFunction<FeatType, LabelType> *lossFunc;
        
    protected:
        string id_str;

    public:
        /**
         * PrintOptInfo print the info of optimization algorithm
         */
        virtual void PrintOptInfo() const {
            printf("--------------------------------------------------\n");
            printf("Algorithm: %s\n\n",this->Id_Str().c_str());
            printf("Learning Rate: %.2f\n", this->eta0);
            printf("Initial t  : %lu\n",this->initial_t);
            printf("Power t : %.2f\n",this->power_t); 
			printf("lambda	: %.2f\n\n",this->lambda);
        }

	public:
		Optimizer(DataSet<FeatType, LabelType> &dataset, LossFunction<FeatType, LabelType> &lossFunc);

		virtual ~Optimizer() {
		}
        const string& Id_Str() const {return this->id_str;}

	protected:
		//Reset the optimizer to the initialization status of training
		virtual void BeginTrain();
		//called when a train ends
		virtual void EndTrain();
		//train the data
		float Train();
		//predict a new feature
		float Predict(const DataPoint<FeatType, LabelType> &data);
		//predict function for test, as we are using sparse learning,dimension of the test data
        //may be larger than the model
		float Test_Predict(const DataPoint<FeatType, LabelType> &data);

		//this is the core of different updating algorithms
		//return the predict
		virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x) = 0;

	public:
		void SetNormalize(bool is_norm){
			this->is_normalize = is_norm;
		}

        void SetParameter(float lambda = -1, float eta0 = -1, 
                float power_t = -1, size_t t0 = 0);
		//try and get the best parameter
		virtual void BestParameter(); 

	public:
		//learn a model
		inline float Learn(int numOfTimes = 1);
		//learn a model and return the mistake rate and its variance
		float Learn(float &aveErrRate, float &varErrRate, float &sparseRate, int numOfTimes = 1);
		//test the performance on the given set
		float Test(DataSet<FeatType, LabelType> &testSet);

		float GetSparseRate(IndexType total_len = 0);

	protected:
		//Change the dimension of weights
		virtual void UpdateWeightSize(IndexType newDim);
    };
    
    //calculate learning rate
    inline float pEta_general(size_t t, float pt){
        return powf((float)t,pt);
    }
    inline float pEta_sqrt(size_t t, float pt){
        return sqrtf((float)t);
    }
    inline float pEta_linear(size_t t, float pt){
        return (float)t;
    }
    inline float pEta_const(size_t t, float pt){
        return 1;
    }

    template <typename FeatType, typename LabelType>
        Optimizer<FeatType, LabelType>::Optimizer(DataSet<FeatType, LabelType> &dataset, 
                LossFunction<FeatType, LabelType> &lossFunc): dataSet(dataset) {
            this->lossFunc = &lossFunc;
            this->weightDim = 1;
            //weight vector
			this->weightVec.resize(this->weightDim);

            this->eta0 = init_eta;;
            this->lambda = init_lambda;
            this->curIterNum = 0;
            this->initial_t = init_initial_t;
            this->power_t = init_power_t;

            this->sparse_soft_thresh = init_sparse_soft_thresh;
			this->is_normalize = false;
        }

    //reset the optimizer to this initialization
    template <typename FeatType, typename LabelType>
        void Optimizer<FeatType, LabelType>::BeginTrain() {
            //reset weight vector
			this->weightVec.set_value(0);
            this->curIterNum = this->initial_t;
        }

    //called when a train ends
    template <typename FeatType, typename LabelType>
        void Optimizer<FeatType, LabelType>::EndTrain() {
            for (IndexType i = 1; i < this->weightDim; i++){
                if (this->weightVec[i] < this->sparse_soft_thresh && 
                        this->weightVec[i] > - this->sparse_soft_thresh ){
                    this->weightVec[i] = 0;
                }
            }
        }

    template <typename FeatType, typename LabelType> 
        float Optimizer<FeatType, LabelType>::Train() {
            if(dataSet.Rewind() == false)
				return 1.f;
            //reset
            this->BeginTrain();
            float errorNum(0);
            size_t show_step = 1; //show information every show_step
            size_t show_count = 2;

            printf("Iterate No.\t\tError Rate\t\t\n");
            while(1) {
                const DataChunk<FeatType,LabelType> &chunk = dataSet.GetChunk();
                //all the data has been processed!
                if(chunk.dataNum  == 0) {
					dataSet.FinishRead();
                    break;
				}

                for (size_t i = 0; i < chunk.dataNum; i++) {
                    const DataPoint<FeatType, LabelType> &data = chunk.data[i];

					IndexType* p_index = data.indexes.begin;
					float* p_feat = data.features.begin;
					if (is_normalize){
						if (data.sum_sq != 1){
							float norm = sqrtf(data.sum_sq);
							while(p_index != data.indexes.end){
								*p_feat /= norm;
								p_index++; p_feat++;
							}
						}
					}

					this->UpdateWeightSize(data.dim());
					float y = this->UpdateWeightVec(data); 
					//loss
					if (this->lossFunc->IsCorrect(data.label,y) == false)
						errorNum++;

					if (show_count == this->curIterNum){
						printf("%lu\t\t\t%.6f\t\t\n",this->curIterNum, 
							errorNum / (float)(this->curIterNum));
						show_count = (1 << ++show_step);
					}
					this->curIterNum++;
				}
				dataSet.FinishRead();
			}
			this->EndTrain();

			return errorNum / dataSet.size();
		}

		//learn a model and return the mistake rate and its variance
		template <typename FeatType, typename LabelType>
		float Optimizer<FeatType, LabelType>::Learn(float &aveErrRate, float &varErrRate, 
			float &sparseRate, int numOfTimes) {
				float * errorRateVec = new float[numOfTimes];
				float * sparseRateVec = new float[numOfTimes];

				for (int i = 0; i < numOfTimes; i++) {
					//random order
					errorRateVec[i] = this->Train();
					sparseRateVec[i] = this->GetSparseRate();
				}
				aveErrRate = Average(errorRateVec, numOfTimes);
				varErrRate = Variance(errorRateVec, numOfTimes);
				sparseRate = Average(sparseRateVec, numOfTimes);

				delete []errorRateVec;
				delete []sparseRateVec;

				return aveErrRate;
		}

		//learn a model
		template <typename FeatType, typename LabelType>
		float Optimizer<FeatType, LabelType>::Learn(int numOfTimes) {
			float aveErrRate, varErrRate, sparseRate;
			return this->Learn(aveErrRate, varErrRate,sparseRate, numOfTimes);
		}

		//test the performance on the given set
		template <typename FeatType, typename LabelType>
		float Optimizer<FeatType, LabelType>::Test(DataSet<FeatType, LabelType> &testSet) {
			if(testSet.Rewind() == false)
				return 1.f;
			float errorRate(0);
			//test
			while(1) {
				const DataChunk<FeatType,LabelType> &chunk = testSet.GetChunk();
				if(chunk.dataNum  == 0) //"all the data has been processed!"
					break;
				for (size_t i = 0; i < chunk.dataNum; i++) {
					const DataPoint<FeatType , LabelType> &data = chunk.data[i];
					IndexType* p_index = data.indexes.begin;
					float* p_feat = data.features.begin;
					if (is_normalize){
						if (data.sum_sq != 1){
							float norm = sqrtf(data.sum_sq);
							while(p_index != data.indexes.end){
								*p_feat /= norm;
								p_index++; p_feat++;
							}
						}
					}
					//predict
					float predict = this->Test_Predict(data);
					if (this->lossFunc->IsCorrect(data.label,predict) == false)
						errorRate++;
				}
				testSet.FinishRead();
			}
			errorRate /= testSet.size();
			return errorRate;
		}

		template <typename FeatType, typename LabelType>
		float Optimizer<FeatType, LabelType>::Test_Predict(const DataPoint<FeatType, LabelType> &data) {
			float predict = 0;
			int dim = data.indexes.size();
			for (int i = 0; i < dim; i++){
				if (data.indexes[i] < this->weightDim)
					predict += this->weightVec[data.indexes[i]] * data.features[i];
			}
			predict += this->weightVec[0];
			return predict;
		}
		template <typename FeatType, typename LabelType>
		float Optimizer<FeatType, LabelType>::Predict(const DataPoint<FeatType, LabelType> &data) {
			float predict = 0;
			IndexType* p_index = data.indexes.begin;
			float* p_feat = data.features.begin;
			while(p_index != data.indexes.end){
				predict += this->weightVec[*p_index++] * (*p_feat++);
			}
			predict += this->weightVec[0];
			return predict;
		}


		template <typename FeatType, typename LabelType>
		float Optimizer<FeatType, LabelType>::GetSparseRate(IndexType total_len) {
			float zeroNum(0);
			if (this->weightDim == 1)
				return 1;

			for (IndexType i = 1; i < this->weightDim; i++) {
				if (this->weightVec[i] == 0)
					zeroNum++;
			}
			if (total_len > 0)
				return zeroNum / total_len;
			else
				return zeroNum / (this->weightDim - 1);
		}

		//try and get the best parameter
		template <typename FeatType, typename LabelType>
		void Optimizer<FeatType, LabelType>::BestParameter() {
			float prev_lambda = this->lambda;
			this->lambda = 0;
			//1. Select the best eta0

			float min_errorRate = 1;
			float bestEta = 1;

			for (float eta_c = init_eta_min; eta_c<= init_eta_max; eta_c *= init_eta_step) {
				cout<<"eta0 = "<<eta_c<<"\n";
				float errorRate(0);
				this->eta0 = eta_c;
				errorRate += this->Train();

				if (errorRate < min_errorRate) {
					bestEta = eta_c;
					min_errorRate = errorRate;
				}
				cout<<"mistake rate: "<<errorRate * 100<<" %\n";
			}
			this->eta0 = bestEta;
			this->lambda = prev_lambda;
			cout<<"Best Parameter:\teta = "<<this->eta0<<"\n\n";
		}

		template <typename FeatType, typename LabelType>
		void Optimizer<FeatType, LabelType>::SetParameter(float lambda , float eta0, 
			float power_t,  size_t t0 ){
				this->lambda  = lambda >= 0 ? lambda : this->lambda;
				this->eta0 = eta0 > 0 ? eta0 : this->eta0;
				this->power_t = power_t >= 0 ? power_t : this->power_t;
				this->initial_t = t0 > 0 ? t0: this->initial_t;
		}

		//Change the dimension of weights
		template <typename FeatType, typename LabelType>
		void Optimizer<FeatType, LabelType>::UpdateWeightSize(IndexType newDim) {
			if (newDim < this->weightDim) 
				return;
			else {
				newDim++; //reserve the 0-th
				this->weightVec.reserve(newDim);
				this->weightVec.resize(newDim); 
				//set the new value to zero
				this->weightVec.zeros(this->weightVec.begin + this->weightDim, 
					this->weightVec.end);
				this->weightDim = newDim;
			}
		}
}
