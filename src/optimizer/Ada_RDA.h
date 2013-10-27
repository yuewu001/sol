/*************************************************************************
> File Name: Ada_RDA.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/18 星期日 17:25:54
> Functions: Adaptive RDA
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
        class Ada_RDA: public Optimizer<FeatType, LabelType> {
        public:
            Ada_RDA(DataSet<FeatType, LabelType> &dataset, 
                    LossFunction<FeatType, LabelType> &lossFunc);
            ~Ada_RDA();

        public:
            //set parameters for specific optimizers
            void SetParameterEx(float lambda = -1, float delta = -1, float eta0 = -1);

            //select the best parameters for the model
            //		virtual void BestParameter();


        protected:
            //this is the core of different updating algorithms
            virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);

            //Change the dimension of weights
            virtual void UpdateWeightSize(int newDim);

            //reset
            virtual void BeginTrain();
            //called when a train ends
            virtual void EndTrain();

        protected:
            float delta;
            unsigned int*timeStamp;
            float *s;
            float *u_t;
    };

    template <typename FeatType, typename LabelType>
        Ada_RDA<FeatType, LabelType>::Ada_RDA(DataSet<FeatType, LabelType> &dataset, 
                LossFunction<FeatType, LabelType> &lossFunc):
            Optimizer<FeatType, LabelType>(dataset, lossFunc), 
            timeStamp(NULL), s(NULL), u_t(NULL) {
                this->delta = init_delta;;
                this->timeStamp = new unsigned int[this->weightDim];
                this->s = new float[this->weightDim];
                this->u_t = new float[this->weightDim];

                this->id_str = "Adaptive RDA";
            }

    template <typename FeatType, typename LabelType>
        Ada_RDA<FeatType, LabelType>::~Ada_RDA() {
            if(this->timeStamp != NULL)
                delete []this->timeStamp;
            if (this->s != NULL)
                delete []this->s;
            if (this->u_t != NULL)
                delete []this->u_t;
        }
    //this is the core of different updating algorithms
    template <typename FeatType, typename LabelType>
        float Ada_RDA<FeatType,LabelType>::UpdateWeightVec(
                const DataPoint<FeatType, LabelType> &x) {
            int featDim = x.indexes.size();
            int index_i = 0;

            //obtain w_t
            for (int i = 0; i < featDim; i++) {
                index_i = x.indexes[i];
                //lazy update
                //update s[i]
                float Htii = this->delta + s[index_i];
                this->weightVec[index_i] = -this->eta0 / Htii *
                    trunc_weight(u_t[index_i], this->lambda * (this->curIterNum - 1));
            }

            //predict 
            float y = this->Predict(x);
            //get gradient
            float gt = this->lossFunc->GetGradient(x.label,y);

            float gt_i = 0;
            //update
            for (int i = 0; i < featDim; i++) {
                index_i = x.indexes[i];
                gt_i = gt * x.features[i];

                this->s[index_i] = sqrt(this->s[index_i] * this->s[index_i] + gt_i * gt_i);
                this->u_t[index_i] += gt_i;
            }
            //bias term
            this->s[0] = sqrt(s[0] * s[0] + gt * gt);
            this->u_t[0] += gt;
            float Htii = this->delta + s[0];
            this->weightVec[0] = -u_t[0] * this->eta0 / Htii; 
            return y;
        }

    //reset the optimizer to this initialization
    template <typename FeatType, typename LabelType>
        void Ada_RDA<FeatType, LabelType>::BeginTrain()
        {
            Optimizer<FeatType, LabelType>::BeginTrain();
            //reset time stamp
            memset(this->timeStamp,0,sizeof(unsigned int) * this->weightDim);
            memset(this->s,0,sizeof(float) * this->weightDim);
            memset(this->u_t, 0 ,sizeof(float) * this->weightDim);
        }
    //called when a train ends
    template <typename FeatType, typename LabelType>
        void Ada_RDA<FeatType, LabelType>::EndTrain() {
            for (int i = 1; i < this->weightDim; i++) {
                //lazy update
                //update s[i]
                float Htii = this->delta + s[i];
                this->weightVec[i] = - this->eta0 / Htii * 
                    trunc_weight(u_t[i], this->lambda * this->curIterNum);
            }
            Optimizer<FeatType,LabelType>::EndTrain();
        }

    /*
    //get the best model parameter
    template <typename FeatType, typename LabelType>
    void Ada_RDA<FeatType, LabelType>::BestParameter()
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
        void Ada_RDA<FeatType, LabelType>::SetParameterEx(float lambda , float delta , float eta0 )
        {
            this->lambda = lambda >= 0 ? lambda : this->lambda;
            this->delta = delta >= 0 ? delta : this->delta;
            this->eta0 = eta0 > 0 ? eta0 : this->eta0;
        }

    //Change the dimension of weights
    template <typename FeatType, typename LabelType>
        void Ada_RDA<FeatType, LabelType>::UpdateWeightSize(int newDim) {
            if (newDim < this->weightDim)
                return;
            else {
                newDim++;
                unsigned int* newT = new unsigned int[newDim];
                float* newS = new float[newDim + 1];
                float* newUt = new float[newDim + 1];
                //copy info
                memcpy(newT,this->timeStamp,sizeof(unsigned int) * this->weightDim);
                memcpy(newS,this->s,sizeof(float) * this->weightDim);
                memcpy(newUt,this->u_t,sizeof(float) * this->weightDim);
                //set the rest to zero
                memset(newT + this->weightDim,0,sizeof(unsigned int) * (newDim - this->weightDim));
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
