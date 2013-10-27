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
            double* w_t1; //w_(t+1)

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
            Optimizer<FeatType, LabelType>(dataset, lossFunc) ,B(NULL), w_t1(NULL) {
        this->id_str = "SGD_QN";

        this->t0 = init_t0;
        this->r = init_r;
        this->skip = init_skip;
        this->B = new double[this->weightDim];
        this->w_t1 = new double[this->weightDim];

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
                int featDim = x.size();
                double y' = this->Predict(x);
                double y = x.label;
                double z = y * y';
                double eta = 1.0 / (this->curIterNum + this->t0) ;
                if(updateB==true) {
                    //update w_t
                    double gt_i = this->lossFunc->GetGradient(x,y);
                    for (int i = 0; i < featDim;i++){
                        this->w_t1[x.indexes[i]] 
                    }
                        FVector w_1=w;
                        double loss_1 = dloss(z);   
                        w.add(x, eta*loss_1*y, Bc);

                        double z2 = y * dot(w,x);
                        double diffloss = dloss(z2) - loss_1;  
                        if (diffloss)
                        {
                            B.compute_ratio(x, lambda, w_1, w, y*diffloss);
                            if(t>skip)
                                Bc.combine_and_clip((t-skip)/(t+skip),B,2*skip/(t+skip),
                                        1/(100*lambda),100/lambda);
                            else
                                Bc.combine_and_clip(t/(t+skip),B,skip/(t+skip),
                                        1/(100*lambda),100/lambda);
                            B.clear();
                            B.resize(w.size());
                        }
                    }
                    updateB=false;    
                }
                else
                {
                    if(--count <= 0)
                    {
                        w.add(w,-skip*lambda*eta,Bc);   
                        count = skip;
                        updateB=true;
                    }      
#if LOSS < LOGLOSS
                    if (z < 1)
#endif
                    {
                        w.add(x, eta*dloss(z)*y, Bc);
                    }
                }
                t += 1;
            }

            if (verb)
                cout << prefix << setprecision(6) << "Norm2: " << dot(w,w) << endl;
            //reset the optimizer to this initialization
            template <typename FeatType, typename LabelType>
                void SGD_QN<FeatType, LabelType>::BeginTrain()
                {
                    Optimizer<FeatType, LabelType>::BeginTrain();
                    //reset time stamp
                    memset(this->B,0,sizeof(size_t) * this->weightDim);
                    count = skip;
                    bool updateB = false;
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

                        double * newW = new double[newDim];
                        //copy info
                        memcpy(newW,this->w_t1,sizeof(double) * this->weightDim); 
                        //set the rest to zero
                        memset(newW + this->weightDim,0,sizeof(double) * (newDim - this->weightDim)); 

                        delete []this->w_t1;
                        this->w_t1 = newW;

                        Optimizer<FeatType,LabelType>::UpdateWeightSize(newDim - 1);
                    }
                }
}


