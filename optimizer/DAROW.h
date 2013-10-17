/*************************************************************************
> File Name: Diagonal AROW
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
	class DAROW: public Optimizer<FeatType, LabelType>
	{
        protected:
            double r;
            double* sigma_w;

            size_t* timeStamp;

        public:
            DAROW(DataSet<FeatType, LabelType> &dataset, 
                    LossFunction<FeatType, LabelType> &lossFunc);
            virtual ~DAROW();

        public:
            void SetParameterEx(double lambda = -1,double r = -1);
        protected:
            //this is the core of different updating algorithms
            virtual double UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
            //reset the optimizer to this initialization
            virtual void BeginTrain();
			 //called when a train ends
            virtual void EndTrain();

            //Change the dimension of weights
            virtual void UpdateWeightSize(int newDim);

            //try and get the best parameter
            virtual void BestParameter(){}

    };

    template <typename FeatType, typename LabelType>
        DAROW<FeatType, LabelType>::DAROW(DataSet<FeatType, LabelType> &dataset, 
                LossFunction<FeatType, LabelType> &lossFunc):
            Optimizer<FeatType, LabelType>(dataset, lossFunc) , sigma_w(NULL)
    {
        this->id_str = "DAROW";
        this->r = init_r;
        this->sigma_w = new double[this->weightDim];
        this->timeStamp = new size_t[this->weightDim];
        this->sparse_soft_thresh = 0;
    }

    template <typename FeatType, typename LabelType>
        DAROW<FeatType, LabelType>::~DAROW()
        {
            if(this->sigma_w != NULL)
                delete []this->sigma_w;
            if (this->timeStamp != NULL)
                delete []this->timeStamp;
        }

    //this is the core of different updating algorithms
    //return the predict
    template <typename FeatType, typename LabelType>
        double DAROW<FeatType,LabelType>::UpdateWeightVec(
                const DataPoint<FeatType, LabelType> &x) {
            double y = this->Predict(x);
            //y /= this->curIterNum;
            double alpha_t = 1 - x.label * y;
            if(alpha_t > 0){
                int featDim = x.indexes.size();
                //calculate beta_t
                double beta_t = this->r;
                for (int i = 0; i < featDim; i++){
                    beta_t += x.features[i] * x.features[i] * this->sigma_w[x.indexes[i]];
                }
                beta_t = 1.0 / beta_t;
                alpha_t *= beta_t; 

                for (int i = 0; i < featDim; i++){
                    int index_i = x.indexes[i];
                    //update u_t
                    this->weightVec[index_i] += alpha_t * this->sigma_w[index_i] * x.label * x.features[i];
                    //L1 lazy update
                    int stepK = this->curIterNum - this->timeStamp[index_i];
                    this->timeStamp[index_i] = this->curIterNum;

                    double w_abs = std::abs(this->weightVec[index_i]); 
                    double alpha =  stepK * this->lambda;
                    if (w_abs > alpha)
                        this->weightVec[index_i] = this->weightVec[index_i] -
                            alpha * Sgn(this->weightVec[index_i]);
                    else
                        this->weightVec[index_i] = 0;

                    //update sigma_w
                    this->sigma_w[index_i] -= beta_t * this->sigma_w[index_i] * this->sigma_w[index_i] * x.features[i] * x.features[i];
                }

                //bias term
                this->weightVec[0] += alpha_t * this->sigma_w[0] * x.label;
                this->sigma_w[0] -= beta_t * this->sigma_w[0] * this->sigma_w[0];
            }
            return y;
        }
    //reset the optimizer to this initialization
    template <typename FeatType, typename LabelType>
        void DAROW<FeatType, LabelType>::BeginTrain()
        {
            Optimizer<FeatType, LabelType>::BeginTrain();

            memset(this->timeStamp,0 ,sizeof(size_t) * this->weightDim);
            for (int i = 0; i < this->weightDim; i++)
                this->sigma_w[i] = 1;
        }

		//called when a train ends
    template <typename FeatType, typename LabelType>
        void DAROW<FeatType, LabelType>::EndTrain() {
            double w_abs = 0, alpha = 0;
            for (int index_i = 1; index_i < this->weightDim; index_i++) {
                    //L1 lazy update
                    int stepK = this->curIterNum - this->timeStamp[index_i];
                    if (stepK == 0)
                        continue;
                    this->timeStamp[index_i] = this->curIterNum;

                    double w_abs = std::abs(this->weightVec[index_i]); 
                    double alpha =  stepK * this->lambda;
                    if (w_abs > alpha)
                        this->weightVec[index_i] = this->weightVec[index_i] -
                            alpha * Sgn(this->weightVec[index_i]);
                    else
                        this->weightVec[index_i] = 0;
            }
            Optimizer<FeatType, LabelType>::EndTrain();
        }

    template <typename FeatType, typename LabelType>
        void DAROW<FeatType, LabelType>::SetParameterEx(double lambda,double r) {
            this->lambda  = lambda >= 0 ? lambda : this->lambda;
            this->r = r > 0 ? r : this->r;
        }

    //Change the dimension of weights
    template <typename FeatType, typename LabelType>
        void DAROW<FeatType, LabelType>::UpdateWeightSize(int newDim)
        {
            if (newDim < this->weightDim)
                return;
            else
            {
                newDim++; //reserve the 0-th

                double * newS = new double[newDim];
                //copy info
                memcpy(newS,this->sigma_w,sizeof(double) * this->weightDim); 
                //set the rest to zero
                for (int i = this->weightDim; i < newDim; i++)
                    newS[i] = 1;

                delete []this->sigma_w;
                this->sigma_w= newS;

                size_t* newT = new size_t[newDim];
                //copy info
                memcpy(newT,this->timeStamp,sizeof(size_t) * this->weightDim);
                //set the rest to zero
                memset(newT + this->weightDim,0,sizeof(size_t) * (newDim - this->weightDim));
                delete []this->timeStamp;
                this->timeStamp = newT;

                Optimizer<FeatType,LabelType>::UpdateWeightSize(newDim - 1);
            }
        }
}
