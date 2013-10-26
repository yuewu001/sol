/************************************************************************
  > File Name: RDA_L1.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: 8/19/2013 Monday 1:49:43 PM
  > Functions: Enhanced L1-RDA method
  > Reference:
  Xiao L. Dual averaging methods for regularized stochastic learning 
  and online optimization[J]. The Journal of Machine Learning Research, 
  2010, 9999: 2543-2596.
 ************************************************************************/

#pragma once

#include "Optimizer.h"
#include "../common/util.h"
#include <cmath>

namespace SOL {
    template <typename FeatType, typename LabelType>
        class RDA_L1: public Optimizer<FeatType, LabelType> {
            public:
                RDA_L1(DataSet<FeatType, LabelType> &dataset, LossFunction<FeatType, LabelType> &lossFunc, bool enchance = true);
                ~RDA_L1();

            public:
                void SetParameterEx(float lambda, float rou = -1);
                //try and get the best parameter
                virtual void BestParameter(); 

            protected:
                //this is the core of different updating algorithms
                //return the predict
                virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
                //reset the optimizer to this initialization
                virtual void BeginTrain();
                //called when a train ends
                virtual void EndTrain();
                //Change the dimension of weights
                virtual void UpdateWeightSize(int newDim);

            protected:
                float rou;
                float gamma_rou;
                float * gtVec; //average gradient vector

        };

    template <typename FeatType, typename LabelType>
        RDA_L1<FeatType, LabelType>::RDA_L1(DataSet<FeatType, LabelType> &dataset, LossFunction<FeatType, LabelType> &lossFunc, bool enchance):
            Optimizer<FeatType, LabelType>(dataset, lossFunc) {
                if(enchance == true){
                    this->id_str = "enhanced RDA";
                    this->rou = init_rou;
                }
                else{
                    this->id_str = "RDA";
                    this->rou = 0;
                }
                this->gtVec = new float[this->weightDim];
                //initail_t should be no less than 1,for the safety of update at the first step
                this->initial_t = this->initial_t < 1 ? 1 : this->initial_t;
            }


    template <typename FeatType, typename LabelType>
        RDA_L1<FeatType,LabelType>::~RDA_L1() {
            if (this->gtVec != NULL)
                delete []this->gtVec;
        }

    template <typename FeatType, typename LabelType>
        float RDA_L1<FeatType,LabelType>::UpdateWeightVec(
                const DataPoint<FeatType, LabelType> &x) {
            size_t featDim = x.indexes.size();
            int index_i = 0;
            //obtain w_t
            float coeff1 = std::sqrt(this->curIterNum - 1);
            float coeff = -this->eta0 / coeff1;
            float lambda_t = this->lambda * (this->curIterNum - 1);
            if (this->rou > 0){
                lambda_t += this->gamma_rou * coeff1;
            }

            for (size_t i = 0; i < featDim; i++) {
                index_i = x.indexes[i];
                this->weightVec[index_i] = coeff * 
                    trunc_weight(this->gtVec[index_i],lambda_t);
            }

            //predict
            float y = this->Predict(x);
            float gt_i = this->lossFunc->GetGradient(x.label,y);

            //update the coeffs
            for (size_t i = 0; i < featDim; i++)
                this->gtVec[x.indexes[i]] += gt_i * x.features[i];

            //bias term
            this->gtVec[0] += gt_i;
            this->weightVec[0] = -this->eta0  * this->gtVec[0] / std::sqrt((float)this->curIterNum);

            return y;
        }

    //
    //reset the optimizer to this initialization
    template <typename FeatType, typename LabelType>
        void RDA_L1<FeatType, LabelType>::BeginTrain() {
            Optimizer<FeatType, LabelType>::BeginTrain();
            memset(this->gtVec, 0, sizeof(float) * this->weightDim);
            this->gamma_rou = this->rou / this->eta0;
        }

    //called when a train ends
    template <typename FeatType, typename LabelType>
        void RDA_L1<FeatType, LabelType>::EndTrain() {
            if (this->curIterNum == 0)
                return;
            float coeff1 = std::sqrt(this->curIterNum);
            float coeff = -this->eta0 / coeff1;
            float lambda_t = this->lambda * (this->curIterNum);
            if (this->rou > 0){
                lambda_t += this->gamma_rou * coeff1;
            }
            for (int index_i = 1; index_i < this->weightDim; index_i++) {
                this->weightVec[index_i] = coeff * trunc_weight(this->gtVec[index_i],
                        lambda_t);
            }

            Optimizer<FeatType, LabelType>::EndTrain();
        }

    template <typename FeatType, typename LabelType>
        void RDA_L1<FeatType,LabelType>::SetParameterEx(float lambda, float rou) {
            this->rou = rou >= 0 ? rou : this->rou;
            this->lambda = lambda >= 0 ? lambda : this->lambda;
        }

    //Change the dimension of weights
    template <typename FeatType, typename LabelType>
        void RDA_L1<FeatType, LabelType>::UpdateWeightSize(int newDim) {
            if (newDim < this->weightDim)
                return;
            else {
                newDim++;
                float* newT = new float[newDim];
                //copy info
                memcpy(newT,this->gtVec,sizeof(float) * this->weightDim);
                //set the rest to zero
                memset(newT + this->weightDim,0,sizeof(float) * (newDim - this->weightDim));
                delete []this->gtVec;
                this->gtVec = newT;

                Optimizer<FeatType,LabelType>::UpdateWeightSize(newDim - 1);
            }
        }

    //try and get the best parameter
    template <typename FeatType, typename LabelType>
        void RDA_L1<FeatType, LabelType>::BestParameter() {
            Optimizer<FeatType,LabelType>::BestParameter();
        }
}
