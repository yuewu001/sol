/*************************************************************************
  > File Name: Sparse Diagonal AROW
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: 2013/8/18 星期日 17:25:54
  > Functions: Diagonal Adaptive Regularization of Weight Vectors
  > Reference: 
  Crammer, Koby, Alex Kulesza, and Mark Dredze. "Adaptive regularization 
  of weight vectors." Machine Learning (2009): 1-33.
 ************************************************************************/

#pragma once


#include "../common/util.h"
#include "Optimizer.h"
#include "HeapList.h"
#include <algorithm>
#include <math.h>
#include <vector>

namespace SOL {
    template <typename FeatType, typename LabelType>
        class ASAROW: public Optimizer<FeatType, LabelType> {
            protected:
                float r;
                float* sigma_w;

                HeapList<float> heap;

                IndexType K; //keep top K elemetns

            public:
                ASAROW(DataSet<FeatType, LabelType> &dataset, 
                        LossFunction<FeatType, LabelType> &lossFunc);
                virtual ~ASAROW();

            public:
                void SetParameterEx(int k = -1, float lambda = -1,float r = -1);
            protected:
                //this is the core of different updating algorithms
                virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
                //reset the optimizer to this initialization
                virtual void BeginTrain();
                //called when a train ends
                virtual void EndTrain();

                //Change the dimension of weights
                virtual void UpdateWeightSize(IndexType newDim);

                //try and get the best parameter
                virtual void BestParameter(){}

        };

    template <typename FeatType, typename LabelType>
        ASAROW<FeatType, LabelType>::ASAROW(DataSet<FeatType, LabelType> &dataset, 
                LossFunction<FeatType, LabelType> &lossFunc):
            Optimizer<FeatType, LabelType>(dataset, lossFunc) , sigma_w(NULL) {
                this->id_str = "ASAROW";
                this->r = init_r;
                this->K = 0;
                this->sigma_w = new float[this->weightDim];
                this->sparse_soft_thresh = 0;
            }

    template <typename FeatType, typename LabelType>
        ASAROW<FeatType, LabelType>::~ASAROW() {
            if(this->sigma_w != NULL)
                delete []this->sigma_w;
        }

    //this is the core of different updating algorithms
    //return the predict
    template <typename FeatType, typename LabelType>
        float ASAROW<FeatType,LabelType>::UpdateWeightVec(
                const DataPoint<FeatType, LabelType> &x) {
            float y = this->Predict(x);
            //y /= this->curIterNum;
            float alpha_t = 1 - x.label * y;
            if(alpha_t > 0){
                size_t featDim = x.indexes.size();
                //calculate beta_t
                float beta_t = this->r;
                for (size_t i = 0; i < featDim; i++){
                    beta_t += x.features[i] * x.features[i] * this->sigma_w[x.indexes[i]];
                }
                beta_t = 1.0 / beta_t;
                alpha_t *= beta_t; 

                for (size_t i = 0; i < featDim; i++){
                    IndexType index_i = x.indexes[i];
                    //update u_t
                    this->weightVec[index_i] += alpha_t * this->sigma_w[index_i] * x.label * x.features[i];
                    //update sigma_w
                    this->sigma_w[index_i] -= beta_t * this->sigma_w[index_i] * this->sigma_w[index_i] * x.features[i] * x.features[i];

                    IndexType ret_id;
                    if(this->heap.UpdateHeap(index_i - 1, ret_id) == true){
                        assert(ret_id + 1 < this->weightDim);
                        this->weightVec[ret_id + 1] = 0; 
                    }
                    //heap.Output(); 
                }
                //bias term
                this->weightVec[0] += alpha_t * this->sigma_w[0] * x.label;
                this->sigma_w[0] -= beta_t * this->sigma_w[0] * this->sigma_w[0];
            }
            return y;
        }
    //reset the optimizer to this initialization
    template <typename FeatType, typename LabelType>
        void ASAROW<FeatType, LabelType>::BeginTrain() {
            if (this->K < 1){
                cerr<<"Please specify a valid number of weights to keep!\n";
                cerr<<"current number: "<<this->K<<endl;
                exit(0);
            }
            Optimizer<FeatType, LabelType>::BeginTrain();
            if (this->weightDim < this->K + 1){
                this->UpdateWeightSize(this->K); //remove the bais term
            }

            for (IndexType i = 0; i < this->weightDim; i++){
                this->sigma_w[i] = 1;
            }
            heap.Init(this->weightDim - 1, this->K, this->sigma_w + 1);
        }

    //called when a train ends
    template <typename FeatType, typename LabelType>
        void ASAROW<FeatType, LabelType>::EndTrain() {
            Optimizer<FeatType, LabelType>::EndTrain();
        }

    template <typename FeatType, typename LabelType>
        void ASAROW<FeatType, LabelType>::SetParameterEx(int k, float lambda,float r) {
            this->K = k > 0 ? k : this->K; 
            this->lambda  = lambda >= 0 ? lambda : this->lambda;
            this->r = r > 0 ? r : this->r;
        }

    //Change the dimension of weights
    template <typename FeatType, typename LabelType>
        void ASAROW<FeatType, LabelType>::UpdateWeightSize(IndexType newDim) {
            if (newDim < this->weightDim)
                return;
            else {
                newDim++; //reserve the 0-th
                float * newS = new float[newDim];
                //copy info
                memcpy(newS,this->sigma_w,sizeof(float) * this->weightDim); 
                //set the rest to zero
                for (IndexType i = this->weightDim; i < newDim; i++)
                    newS[i] = 1;

                delete []this->sigma_w;
                this->sigma_w= newS;

                heap.UpdateDataNum(newDim - 1, this->sigma_w + 1);
                Optimizer<FeatType,LabelType>::UpdateWeightSize(newDim - 1);
            }
        }
}
