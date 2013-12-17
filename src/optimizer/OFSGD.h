/*************************************************************************
  > File Name: OFSGD.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Tue 17 Dec 2013 04:59:06 AM EST
  > Descriptions: Online Feature Selection
  > Reference: Online Feature Selection and its applications
 ************************************************************************/
#ifndef HEADER_OPTIMISER_OFSGD
#define HEADER_OPTIMISER_OFSGD

#include "../common/util.h"
#include "Optimizer.h"
#include "HeapList.h"
#include <algorithm>
#include <math.h>
#include <vector>

namespace SOL {
    template <typename FeatType, typename LabelType>
        class OFSGD: public Optimizer<FeatType, LabelType> {
            protected:
                HeapList<float> heap;

                IndexType K; //keep top K elemetns

                float w_norm;
                float norm_coeff;

                float (*pEta_time)(size_t t, float pt);

            public:
                OFSGD(DataSet<FeatType, LabelType> &dataset, 
                        LossFunction<FeatType, LabelType> &lossFunc);
                virtual ~OFSGD();

            public:
                void SetParameterEx(int k);
                /**
                 * PrintOptInfo print the info of optimization algorithm
                 */
                virtual void PrintOptInfo() const {
                    printf("--------------------------------------------------\n");
                    printf("Algorithm: %s\n\n",this->Id_Str().c_str());
                    printf("eta:\t%.2f\n", this->eta);
                    printf("lambda:\t%.2f\n\n", this->lambda);
                }
            protected:
                //this is the core of different updating algorithms
                virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);
                //reset the optimizer to this initialization
                virtual void BeginTrain();
                //called when a train ends
                virtual void EndTrain();

                //Change the dimension of weights
                virtual void UpdateWeightSize(IndexType newDim);
        };

    template <typename FeatType, typename LabelType>
        OFSGD<FeatType, LabelType>::OFSGD(DataSet<FeatType, LabelType> &dataset, 
                LossFunction<FeatType, LabelType> &lossFunc):
            Optimizer<FeatType, LabelType>(dataset, lossFunc){
                this->id_str = "OFSGD";
                this->K = 0;
            }

    template <typename FeatType, typename LabelType>
        OFSGD<FeatType, LabelType>::~OFSGD() {
        }

    //this is the core of different updating algorithms
    //return the predict
    template <typename FeatType, typename LabelType>
        float OFSGD<FeatType,LabelType>::UpdateWeightVec(
                const DataPoint<FeatType, LabelType> &x) {
            //we use the oposite of w
            float y = -this->Predict(x);
            size_t featDim = x.indexes.size();
            float gt_i = this->lossFunc->GetGradient(x.label,y);
            if (gt_i == 0)
                return y;

            this->eta = this->eta0 / this->pEta_time(this->curIterNum, this->power_t);

            IndexType index_i = 0;
            for (size_t i = 0; i < featDim; i++) {
                index_i = x.indexes[i];
                w_norm -= this->weightVec[index_i] * this->weightVec[index_i];
                this->weightVec[index_i] += this->eta * gt_i * x.features[i];
                w_norm += this->weightVec[index_i] * this->weightVec[index_i];

                IndexType ret_id;
                this->heap.UpdateHeap(index_i - 1, ret_id); 
                if (this->heap.is_topK(index_i - 1)){
                    this->weightVec[ret_id + 1] = 0;
                }
                else{
                    this->weightVec[index_i] = 0;
                }
                //heap.Output(); 
            }
            //update bias 
            w_norm -= this->weightVec[0] * this->weightVec[0];
            this->weightVec[0] += this->eta * gt_i;
            w_norm += this->weightVec[0] * this->weightVec[0];

            //normlize
            float coeff = this->norm_coeff / sqrtf(w_norm);
            if (coeff < 1){
                for (IndexType i = 0; i < this->weightDim; i++)
                    this->weightVec[i] *= coeff;

            }
            return y;
        }

    //reset the optimizer to this initialization
    template <typename FeatType, typename LabelType>
        void OFSGD<FeatType, LabelType>::BeginTrain() {
            Optimizer<FeatType, LabelType>::BeginTrain();
            if (this->K < 1){
                cerr<<"Please specify a valid number of weights to keep!\n";
                cerr<<"current number: "<<this->K<<endl;
                exit(0);
            }
            if (this->weightDim < this->K + 1){
                this->UpdateWeightSize(this->K); //remove the bais term
            }
            heap.Init(this->weightDim - 1, this->K, this->weightVec.begin + 1);

            this->w_norm = 0;
            this->norm_coeff = 1.f / sqrtf(this->lambda);

            this->power_t = 0;
            if (this->power_t == 0.5)
                this->pEta_time = pEta_sqrt;
            else if(this->power_t == 0)
                this->pEta_time = pEta_const;
            else if (this->power_t == 1)
                this->pEta_time = pEta_linear;
            else
                this->pEta_time = pEta_general;

        }

    //called when a train ends
    template <typename FeatType, typename LabelType>
        void OFSGD<FeatType, LabelType>::EndTrain() {
            for (IndexType i = 0; i < this->weightDim; i++)
                this->weightVec[i] = -this->weightVec[i];

            Optimizer<FeatType, LabelType>::EndTrain();
        }

    template <typename FeatType, typename LabelType>
        void OFSGD<FeatType, LabelType>::SetParameterEx(int k) {
            if (k < 1){
                cerr<<"Please specify a valid number of weights to keep!\n";
                cerr<<"current number: "<<this->K<<endl;
                exit(0);
            }
            else
                this->K = k;
        }

    //Change the dimension of weights
    template <typename FeatType, typename LabelType>
        void OFSGD<FeatType, LabelType>::UpdateWeightSize(IndexType newDim) {
            if (newDim < this->weightDim)
                return;
            else {
                Optimizer<FeatType,LabelType>::UpdateWeightSize(newDim);
                heap.UpdateDataNum(newDim, this->weightVec.begin + 1);
            }
        }
}

#endif
