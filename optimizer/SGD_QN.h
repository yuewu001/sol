/*************************************************************************
> File Name: SGD_QN.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/18 星期日 17:25:54
> Functions: Sparse Online Learning With Truncated Gradient
> Reference:
************************************************************************/

#pragma once


#include "../common/util.h"
#include "Optimizer.h"
#include <cmath>
#include <limits>

namespace SOL
{
	template <typename FeatType, typename LabelType>
	class SGD_QN: public Optimizer<FeatType, LabelType> {
        private:
            size_t t0; 
            int r;
            bool updateB;
            int count;
            int skip;
            double* B;

        public:
            SGD_QN(DataSet<FeatType, LabelType> &dataset, 
                    LossFunction<FeatType, LabelType> &lossFunc);
            virtual ~SGD_QN();

        public:
            void SetParameterEx(double lambda = -1, int skip = -1,
                   int r0 = -1, double eta = -1);
        protected:
            //this is the core of different updating algorithms
            virtual double UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
            //reset the optimizer to this initialization
            virtual void BeginTrain();
			 //called when a train ends
            virtual void EndTrain();

            //Change the dimension of weights
            virtual void UpdateWeightSize(int newDim);
    };

    template <typename FeatType, typename LabelType>
        SGD_QN<FeatType, LabelType>::SGD_QN(DataSet<FeatType, LabelType> &dataset, 
                LossFunction<FeatType, LabelType> &lossFunc):
            Optimizer<FeatType, LabelType>(dataset, lossFunc) ,B(NULL) {
        this->id_str = "SGD_QN";

        this->t0 = init_t0;
        this->r = init_r;
        this->skip = init_skip;
        this->B = new double[this->weightDim];

        //sparse soft threshold
        this->sparse_soft_thresh = 0;
    }

    template <typename FeatType, typename LabelType>
        SGD_QN<FeatType, LabelType>::~SGD_QN() {
            if(this->B!= NULL)
                delete []this->B;
        }

    //this is the core of different updating algorithms
    //return the predict
    template <typename FeatType, typename LabelType>
        double SGD_QN<FeatType,LabelType>::UpdateWeightVec(
                const DataPoint<FeatType, LabelType> &x) {
            double y = this->Predict(x);
            int featDim = x.indexes.size();
            double gt_i = this->lossFunc->GetGradient(x,y);

            int index_i = 0;
            double w_abs = 0, alpha = 0;
            for (int i = 0; i < featDim; i++)
            {
                index_i = x.indexes[i];
                //update the weight
                this->weightVec[index_i] -= this->eta * gt_i * x.features[i];

                //lazy update
                //truncated gradient
                int stepK = ((this->curIterNum - this->timeStamp[index_i]) / this->K) 
                    * this->K;

                if (this->timeStamp[index_i] == 0)
                {
                    this->timeStamp[index_i] = this->curIterNum;
                    stepK = 0;
                    continue;
                }
                else
                    this->timeStamp[index_i] += stepK;
				if (stepK == 0)
					continue;

                w_abs = std::abs(this->weightVec[index_i]); 
                if (w_abs > this->theta)
                    continue;
                else 
                {
                    alpha =  stepK * this->eta * this->lambda;
                    if (w_abs > alpha)
                        this->weightVec[index_i] -= alpha * Sgn(this->weightVec[index_i]);
                    else
                        this->weightVec[index_i] = 0;
                }
            }

            //bias term
            this->weightVec[0] -= this->eta * gt_i;

            return y;
        }
    //reset the optimizer to this initialization
    template <typename FeatType, typename LabelType>
        void SGD_QN<FeatType, LabelType>::BeginTrain()
        {
            Optimizer<FeatType, LabelType>::BeginTrain();
            //reset time stamp
            memset(this->B,0,sizeof(size_t) * this->weightDim);
        }

		//called when a train ends
    template <typename FeatType, typename LabelType>
        void SGD_QN<FeatType, LabelType>::EndTrain(){
        }
        

    template <typename FeatType, typename LabelType>
        void SGD_QN<FeatType, LabelType>::SetParameterEx(double lambda, 
                int skip, int r0, double eta) {
            this->lambda  = lambda >= 0 ? lambda : this->lambda;
            this->skip =skip > 0 ? skip : this->skip;
            this->r0 = r0 > 0 ? r0 : this->r0;
            this->eta = Eta > 0 ? Eta : this->eta;
        }

    //Change the dimension of weights
    template <typename FeatType, typename LabelType>
        void SGD_QN<FeatType, LabelType>::UpdateWeightSize(int newDim)
        {
            if (newDim < this->weightDim)
                return;
            else
            {
                newDim++; //reserve the 0-th

                double * newB = new double[newDim];
                //copy info
                memcpy(newB,this->B,sizeof(double) * this->weightDim); 
                //set the rest to zero
                memset(newB + this->weightDim,0,sizeof(double) * (newDim - this->weightDim)); 

                delete []this->B;
                this->B = newB;

                Optimizer<FeatType,LabelType>::UpdateWeightSize(newDim - 1);
            }
        }
}


