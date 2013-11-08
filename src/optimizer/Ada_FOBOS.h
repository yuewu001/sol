/*************************************************************************
  > File Name: Ada_FOBOS.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Sat 26 Oct 2013 12:17:04 PM SGT
  > Descriptions: adaptive fobos algorithm
  > Reference:
  Duchi J, Hazan E, Singer Y. Adaptive subgradient methods for 
  online learning and stochastic optimization[J]. The Journal of 
  Machine Learning Research, 2011, 999999: 2121-2159.

  This file implements the L1 regularization
 ************************************************************************/
#pragma once
#include "Optimizer.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace SOL {
    template <typename FeatType, typename LabelType>
        class Ada_FOBOS: public Optimizer<FeatType, LabelType> {
        public:
            Ada_FOBOS(DataSet<FeatType, LabelType> &dataset, 
                    LossFunction<FeatType, LabelType> &lossFunc);
            ~Ada_FOBOS();

        public:
            //set parameters for specific optimizers
            void SetParameterEx(float lambda = -1, float delta = -1, float eta0 = -1);

            //select the best parameters for the model
            //		virtual void BestParameter();


        protected:
            //this is the core of different updating algorithms
            virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);

            //Change the dimension of weights
            virtual void UpdateWeightSize(IndexType newDim);

            //reset
            virtual void BeginTrain();
            //called when a train ends
            virtual void EndTrain();

        protected:
            float delta;
            size_t *timeStamp;
            float *s;
            float *u_t;
    };

    template <typename FeatType, typename LabelType>
        Ada_FOBOS<FeatType, LabelType>::Ada_FOBOS(DataSet<FeatType, LabelType> &dataset, 
                LossFunction<FeatType, LabelType> &lossFunc):
            Optimizer<FeatType, LabelType>(dataset, lossFunc), timeStamp(NULL),
            s(NULL), u_t(NULL) {
                this->delta = init_delta;;
                this->timeStamp = new size_t[this->weightDim];
                this->s = new float[this->weightDim];
                this->u_t = new float[this->weightDim];

                this->id_str = "Adaptive FOBOS";
            }

    template <typename FeatType, typename LabelType>
        Ada_FOBOS<FeatType, LabelType>::~Ada_FOBOS() {
            if(this->timeStamp != NULL)
                delete []this->timeStamp;
            if (this->s != NULL)
                delete []this->s;
            if (this->u_t != NULL)
                delete []this->u_t;
        }
    //update witt Composite Mirror-Descent
    template <typename FeatType, typename LabelType>
        float Ada_FOBOS<FeatType,LabelType>::UpdateWeightVec(
                const DataPoint<FeatType, LabelType> &x) {
            size_t featDim = x.indexes.size();
            IndexType index_i = 0;
            float alpha = this->eta0 * this->lambda;
            for (size_t i = 0; i < featDim; i++) {
                index_i = x.indexes[i];
                //update s[i]
                float Ht0i = this->delta + s[index_i];

                //to obtain w_(t + 1),i, first calculate w_t,i
                this->weightVec[index_i] = trunc_weight(this->weightVec[index_i],
                        alpha * (this->curIterNum - this->timeStamp[index_i]) / Ht0i);

                //update the time stamp
                this->timeStamp[index_i] = this->curIterNum;
            }
            float y = this->Predict(x);
            //get gradient
            float gt = this->lossFunc->GetGradient(x.label,y);
            float gt_i = 0;

            //update s[i]
            for (size_t i = 0; i < featDim; i++) {
                index_i = x.indexes[i];
                gt_i = gt * x.features[i];

                this->s[index_i] = sqrt(s[index_i] * s[index_i] + gt_i * gt_i);
                float Htii = this->delta + s[index_i];
                //obtain w_(t + 1),i
                this->weightVec[index_i] -= this->eta0 * gt_i / Htii;
            }

            //bias term
            this->s[0] = sqrt(s[0] * s[0] + gt * gt);
            float Htii = this->delta + s[0];
            this->weightVec[0] -= this->eta0 * gt / Htii;

            return y;
        }

    //reset the optimizer to this initialization
    template <typename FeatType, typename LabelType>
        void Ada_FOBOS<FeatType, LabelType>::BeginTrain() {
            Optimizer<FeatType, LabelType>::BeginTrain();
            //reset time stamp
            memset(this->timeStamp,0,sizeof(size_t) * this->weightDim);
            memset(this->s,0,sizeof(float) * this->weightDim);
            memset(this->u_t, 0 ,sizeof(float) * this->weightDim);
        }
    //called when a train ends
    template <typename FeatType, typename LabelType>
        void Ada_FOBOS<FeatType, LabelType>::EndTrain() {
            size_t iterNum = this->curIterNum + 1;
            float alpha = 0;
            for (IndexType index_i = 1; index_i < this->weightDim; index_i++) {
                //update s[i]
                float Ht0i = this->delta + s[index_i];
                alpha = this->lambda * this->eta0 * (iterNum - this->timeStamp[index_i]) / Ht0i;
                this->weightVec[index_i] = trunc_weight(this->weightVec[index_i], alpha);
            }

            Optimizer<FeatType,LabelType>::EndTrain();
        }

    /*
    //get the best model parameter
    template <typename FeatType, typename LabelType>
    void Ada_FOBOS<FeatType, LabelType>::BestParameter()
    {
    float prevLambda = this->lambda;
    this->lambda = 0;

    //Select the best eta0
    float min_errorRate = 1;
    float bestEta = 1;
    float bestDelta = 1;

    for (float eta_c = init_eta_min; eta_c<= init_eta_max; eta_c *= init_eta_step)
    {
    this->eta0 = eta_c;
    for (float delt = init_delta_min; delt <= init_delta_max; delt *= init_delta_step)
    {
    cout<<"eta0 = "<<eta_c<<" delta= "<<delt;
    this->delta = delt;
    float errorRate(0);
    errorRate = this->Train();

    if (errorRate < min_errorRate)
    {
    bestEta = eta_c;
    bestDelta = delt;
    min_errorRate = errorRate;
    }
    cout<<" mistake rate: "<<errorRate * 100<<" %\n";
    }
    }

    this->eta0 = bestEta;
    this->delta = bestDelta;
    this->lambda = prevLambda;
    cout<<"Best Parameter:\teta = "<<this->eta0<<"\tdelta = "<<this->delta<<"\n\n";
    }
    */

    //set parameters for specific optimizers
    template <typename FeatType, typename LabelType>
        void Ada_FOBOS<FeatType, LabelType>::SetParameterEx(float lambda , float delta , float eta0 ) {
            this->lambda = lambda >= 0 ? lambda : this->lambda;
            this->delta = delta >= 0 ? delta : this->delta;
            this->eta0 = eta0 > 0 ? eta0 : this->eta0;
        }

    //Change the dimension of weights
    template <typename FeatType, typename LabelType>
        void Ada_FOBOS<FeatType, LabelType>::UpdateWeightSize(IndexType newDim) {
            if (newDim < this->weightDim)
                return;
            else {
                newDim++;
                size_t* newT = new size_t[newDim];
                float* newS = new float[newDim + 1];
                float* newUt = new float[newDim + 1];
                //copy info
                memcpy(newT,this->timeStamp,sizeof(size_t) * this->weightDim);
                memcpy(newS,this->s,sizeof(float) * this->weightDim);
                memcpy(newUt,this->u_t,sizeof(float) * this->weightDim);
                //set the rest to zero
                memset(newT + this->weightDim,0,sizeof(size_t) * (newDim - this->weightDim));
                memset(newS + this->weightDim,0,sizeof(float) * (newDim - this->weightDim));
                memset(newUt + this->weightDim,0,sizeof(float) * (newDim - this->weightDim));

                delete []this->timeStamp;
                delete []this->s;
                delete []this->u_t;

                this->timeStamp = newT;
                this->s = newS;
                this->u_t = newUt;

                Optimizer<FeatType,LabelType>::UpdateWeightSize(newDim - 1);
            }
        }
}

