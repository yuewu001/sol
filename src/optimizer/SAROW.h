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
#include <algorithm>
#include <math.h>
#include <limits>

namespace SOL {
    struct sortItem{
        unsigned int index;
        float sigma;
        friend bool operator < (const sortItem& item1, const sortItem& item2){
            return item1.sigma < item2.sigma;
        }
    };

	template <typename FeatType, typename LabelType>
	class SAROW: public Optimizer<FeatType, LabelType> {
        protected:
            float r;
            float* sigma_w;
            unsigned int* forward_pos_list; //record the item of weight in forward direction
            unsigned int* backward_pos_list; //record the item of weight in backward direction

            int K; //keep top K elemetns

        public:
            SAROW(DataSet<FeatType, LabelType> &dataset, 
                    LossFunction<FeatType, LabelType> &lossFunc);
            virtual ~SAROW();

        public:
            void SetParameterEx(float lambda = -1,float r = -1);
        protected:
            //this is the core of different updating algorithms
            virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
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
        SAROW<FeatType, LabelType>::SAROW(DataSet<FeatType, LabelType> &dataset, 
                LossFunction<FeatType, LabelType> &lossFunc):
            Optimizer<FeatType, LabelType>(dataset, lossFunc) , sigma_w(NULL) {
        this->id_str = "SAROW";
        this->r = init_r;
        this->sigma_w = new float[this->weightDim];
        this->forward_pos_list = new unsigned int[this->weightDim];
        this->backward_pos_list = new unsigned int[this->weightDim];
        this->sparse_soft_thresh = 0;
    }

    template <typename FeatType, typename LabelType>
        SAROW<FeatType, LabelType>::~SAROW() {
            if(this->sigma_w != NULL)
                delete []this->sigma_w;
        }

    //this is the core of different updating algorithms
    //return the predict
    template <typename FeatType, typename LabelType>
        float SAROW<FeatType,LabelType>::UpdateWeightVec(
                const DataPoint<FeatType, LabelType> &x) {
            float y = this->Predict(x);
            //y /= this->curIterNum;
            float alpha_t = 1 - x.label * y;
            if(alpha_t > 0){
                int featDim = x.indexes.size();
                //calculate beta_t
                float beta_t = this->r;
                for (int i = 0; i < featDim; i++){
                    beta_t += x.features[i] * x.features[i] * this->sigma_w[x.indexes[i]];
                }
                beta_t = 1.0 / beta_t;
                alpha_t *= beta_t; 

                for (int i = 0; i < featDim; i++){
                    int index_i = x.indexes[i];
                    //update u_t
                    this->weightVec[index_i] += alpha_t * this->sigma_w[index_i] * x.label * x.features[i];
                    //update sigma_w
                    this->sigma_w[index_i] -= beta_t * this->sigma_w[index_i] * this->sigma_w[index_i] * x.features[i] * x.features[i];

                    //update the forward backward position
                    unsigned int new_pos = this->backward_pos_list[index_i];
                    while(new_pos > 1){
                        unsigned int for_pos = this->forward_pos_list[new_pos - 1];
                        if (this->sigma_w[for_pos] <= 
                                this->sigma_w[index_i]){
                            break;
                        } 
                        else{
                            this->forward_pos_list[new_pos] = for_pos;
                            this->backward_pos_list[for_pos] = new_pos;
                            new_pos--;
                        }
                    }
                    this->forward_pos_list[new_pos] = index_i;
                    this->backward_pos_list[index_i] = new_pos;
                }

                //truncate
                for (int i = this->K + 1; i < this->weightDim; i++){
                    this->weightVec[this->forward_pos_list[i]] = 0;
                }

                //bias term
                this->weightVec[0] += alpha_t * this->sigma_w[0] * x.label;
                this->sigma_w[0] -= beta_t * this->sigma_w[0] * this->sigma_w[0];
            }
            return y;
        }
    //reset the optimizer to this initialization
    template <typename FeatType, typename LabelType>
        void SAROW<FeatType, LabelType>::BeginTrain() {
            Optimizer<FeatType, LabelType>::BeginTrain();

            for (int i = 0; i < this->weightDim; i++){
                this->sigma_w[i] = 1;
                this->forward_pos_list[i] = i;
                this->backward_pos_list[i] = i;
            }
            this->K = this->weightDim * (this->lambda < 1 ? this->lambda : 1);
        }

		//called when a train ends
    template <typename FeatType, typename LabelType>
        void SAROW<FeatType, LabelType>::EndTrain() {
            for (int index_i = 1; index_i < this->weightDim; index_i++) {
            }
            Optimizer<FeatType, LabelType>::EndTrain();
        }

    template <typename FeatType, typename LabelType>
        void SAROW<FeatType, LabelType>::SetParameterEx(float lambda,float r) {
            this->lambda  = lambda >= 0 ? lambda : this->lambda;
            this->r = r > 0 ? r : this->r;
        }

    //Change the dimension of weights
    template <typename FeatType, typename LabelType>
        void SAROW<FeatType, LabelType>::UpdateWeightSize(int newDim) {
            if (newDim < this->weightDim)
                return;
            else {
                newDim++; //reserve the 0-th
                float * newS = new float[newDim];
                //copy info
                memcpy(newS,this->sigma_w,sizeof(float) * this->weightDim); 
                //set the rest to zero
                for (int i = this->weightDim; i < newDim; i++)
                    newS[i] = 1;

                delete []this->sigma_w;
                this->sigma_w= newS;

                unsigned int* newP = new unsigned int[newDim];
                //copy info
                memcpy(newP, this->forward_pos_list, sizeof(unsigned int) * this->weightDim);
                //set the rest 
                for (int i = this->weightDim; i < newDim; i++){
                    newP[i] = i;
                }
                delete []this->forward_pos_list;
                this->forward_pos_list = newP;

                newP = new unsigned int[newDim];
                //copy info
                memcpy(newP, this->backward_pos_list, sizeof(unsigned int) * this->weightDim);
                //set the rest 
                for (int i = this->weightDim; i < newDim; i++){
                    newP[i] = i;
                }
                delete []this->backward_pos_list;
                this->backward_pos_list = newP;

                this->K = newDim * (this->lambda < 1 ? this->lambda : 1);
                Optimizer<FeatType,LabelType>::UpdateWeightSize(newDim - 1);
            }
        }
}
