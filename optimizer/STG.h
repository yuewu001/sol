/*************************************************************************
> File Name: STG.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/18 星期日 17:25:54
> Functions: Sparse Online Learning With Truncated Gradient
> Reference:
Langford J, Li L, Zhang T. Sparse online learning via truncated 
gradient[J]. The Journal of Machine Learning Research, 2009, 10: 
777-801. 
************************************************************************/

#pragma once


#include "../common/util.h"
#include "Optimizer.h"
#include <math.h>
#include <limits>

namespace SOL {
	template <typename FeatType, typename LabelType>
	class STG: public Optimizer<FeatType, LabelType> {
        protected:
            int K;

        protected:
            unsigned int *timeStamp;

        public:
            STG(DataSet<FeatType, LabelType> &dataset, 
                    LossFunction<FeatType, LabelType> &lossFunc);
            virtual ~STG();

        public:
            void SetParameterEx(float lambda = -1,int K = -1, 
                    float eta = -1);
        protected:
            //this is the core of different updating algorithms
            virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
            //reset the optimizer to this initialization
            virtual void BeginTrain();
			 //called when a train ends
            virtual void EndTrain();

            //Change the dimension of weights
            virtual void UpdateWeightSize(int newDim);
    };

    template <typename FeatType, typename LabelType>
        STG<FeatType, LabelType>::STG(DataSet<FeatType, LabelType> &dataset, 
                LossFunction<FeatType, LabelType> &lossFunc):
            Optimizer<FeatType, LabelType>(dataset, lossFunc) , timeStamp(NULL) {
        this->id_str = "STG";
        this->K = init_k;
        this->timeStamp = new unsigned int[this->weightDim];
		if (this->K != 1)
			this->sparse_soft_thresh = init_sparse_soft_thresh;
		else
			this->sparse_soft_thresh = 0;
    }

    template <typename FeatType, typename LabelType>
        STG<FeatType, LabelType>::~STG() {
            if(this->timeStamp != NULL)
                delete []this->timeStamp;
        }

    //this is the core of different updating algorithms
    //return the predict
    template <typename FeatType, typename LabelType>
        float STG<FeatType,LabelType>::UpdateWeightVec(const DataPoint<FeatType, LabelType> &x) {
            float y = this->Predict(x);
            int featDim = x.indexes.size();
            float gt_i = this->lossFunc->GetGradient(x.label,y) * this->eta;

            float alpha = this->eta * this->lambda;
            unsigned int stepK = 0;

            for (int i = 0; i < featDim; i++) {
                const int &index_i = x.indexes[i];
                //update the weight
                float& p_weight = this->weightVec[index_i];
                unsigned int & p_stamp = this->timeStamp[index_i];
                p_weight -= gt_i * x.features[i];

                //lazy update
                //truncated gradient
                if (p_stamp == 0) {
                    p_stamp = this->curIterNum;
                    continue;
                }
                else{
                    stepK = this->curIterNum - p_stamp;
                    stepK -= stepK % this->K;
                    p_stamp += stepK;
                }

                p_weight = trunc_weight(p_weight,stepK * alpha);
            }

            //bias term
            this->weightVec[0] -= gt_i;

            return y;
        }

    
    //reset the optimizer to this initialization
    template <typename FeatType, typename LabelType>
        void STG<FeatType, LabelType>::BeginTrain() {
            Optimizer<FeatType, LabelType>::BeginTrain();
            //reset time stamp
            memset(this->timeStamp,0,sizeof(unsigned int) * this->weightDim);
        }

		//called when a train ends
    template <typename FeatType, typename LabelType>
        void STG<FeatType, LabelType>::EndTrain() {
			float w_abs = 0, alpha = 0;
			for (int index_i = 1; index_i < this->weightDim; index_i++) {
				//truncated gradient
				int stepK = this->curIterNum - this->timeStamp[index_i];
                stepK -= stepK % this->K;

				if (stepK == 0)
					continue;

                this->weightVec[index_i] = trunc_weight(this->weightVec[index_i],
                        stepK * this->eta * this->lambda);
			}
            Optimizer<FeatType, LabelType>::EndTrain();
        }

    template <typename FeatType, typename LabelType>
        void STG<FeatType, LabelType>::SetParameterEx(float lambda , int k, float Eta) {
            this->lambda  = lambda >= 0 ? lambda : this->lambda;
            this->K = k > 0 ? k : this->K;
            this->eta = Eta > 0 ? Eta : this->eta;
			if (this->K == 1)
				this->sparse_soft_thresh = 0;
			else
				this->sparse_soft_thresh = init_sparse_soft_thresh;
        }

    //Change the dimension of weights
    template <typename FeatType, typename LabelType>
        void STG<FeatType, LabelType>::UpdateWeightSize(int newDim) {
            if (newDim < this->weightDim)
                return;
            else {
                newDim++; //reserve the 0-th
                unsigned int* newT = new unsigned int[newDim];
                //copy info
                memcpy(newT,this->timeStamp,sizeof(unsigned int) * this->weightDim); 
                //set the rest to zero
                memset(newT + this->weightDim,0,sizeof(unsigned int) * (newDim - this->weightDim)); 

                delete []this->timeStamp;
                this->timeStamp = newT;

                Optimizer<FeatType,LabelType>::UpdateWeightSize(newDim - 1);
            }
        }
}
