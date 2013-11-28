/*************************************************************************
  > File Name: CW_RDA.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: 2013/11/27 14:04:06
  > Functions: Confidence weighted regularized dual averaging
 ************************************************************************/
#ifndef HEADER_CW_RDA
#define HEADER_CW_RDA

#include "Optimizer.h"
#include <cmath>
#include <stdexcept>

namespace SOL {
    template <typename FeatType, typename LabelType>
        class CW_RDA: public Optimizer<FeatType, LabelType> {
            protected:
                float r;
                s_array<float> sigma_w;
                s_array<float> u_t;
				float gravity;

            public:
                CW_RDA(DataSet<FeatType, LabelType> &dataset, 
                        LossFunction<FeatType, LabelType> &lossFunc);
                ~CW_RDA();

            public:
                //set parameters for specific optimizers
                void SetParameterEx(float r = -1);

            protected:
                //this is the core of different updating algorithms
                virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);

                //Change the dimension of weights
                virtual void UpdateWeightSize(IndexType newDim);

                //reset
                virtual void BeginTrain();
                //called when a train ends
                virtual void EndTrain();

                //try and get the best parameter
                virtual void BestParameter(){}

        };

    template <typename FeatType, typename LabelType>
        CW_RDA<FeatType, LabelType>::CW_RDA(DataSet<FeatType, LabelType> &dataset, 
                LossFunction<FeatType, LabelType> &lossFunc):
            Optimizer<FeatType, LabelType>(dataset, lossFunc){
                this->id_str = "Confidence Weighted RDA";
                this->r = init_r;
                this->u_t.resize(this->weightDim);
                this->sigma_w.resize(this->weightDim);
                this->lossFunc = new SquaredHingeLoss<FeatType, LabelType>;
            }

    template <typename FeatType, typename LabelType>
	CW_RDA<FeatType, LabelType>::~CW_RDA() {
		if (this->lossFunc != NULL){
			delete this->lossFunc;
			this->lossFunc = NULL;
		}
	}
    //this is the core of different updating algorithms
    template <typename FeatType, typename LabelType>
        float CW_RDA<FeatType,LabelType>::UpdateWeightVec(
                const DataPoint<FeatType, LabelType> &x) {
            size_t featDim = x.indexes.size();
			IndexType* p_index = x.indexes.begin;
			float* p_feat = x.features.begin;
			//obtain w_t
			while(p_index != x.indexes.end){
				//lazy update
				this->weightVec[*p_index] = -this->sigma_w[*p_index] *
					trunc_weight(u_t[*p_index], gravity * (this->curIterNum - 1));
				p_index++;
			}

			//predict 
			float y = this->Predict(x);
			//get gradient
			float gt = this->lossFunc->GetGradient(x.label,y);
			if (gt != 0){
				//calculate learning rate
				this->eta = this->r;
				float temp_sum = 0;
				p_index = x.indexes.begin;
				p_feat = x.features.begin;
				while(p_index != x.indexes.end){
					temp_sum = (*p_feat) * (*p_feat) * this->sigma_w[*p_index];
					this->eta += temp_sum;
					//update sigma_w
					this->sigma_w[*p_index] *= this->r / (this->r + temp_sum);
					p_index++;p_feat++;
				}

				this->eta = 0.5f / this->eta;
				gravity = this->lambda * this->eta;
				gt *= this->eta;
				//update
				p_index = x.indexes.begin;
				p_feat = x.features.begin;
				while(p_index != x.indexes.end){
					this->u_t[*p_index++] += gt * (*p_feat++);
				}

				//bias term
				this->u_t[0] += gt;
				this->weightVec[0] = -u_t[0] * this->sigma_w[0]; 
				this->sigma_w[0] *= this->r / (this->r + this->sigma_w[0]);
			}
			return y;
		}

		//reset the optimizer to this initialization
		template <typename FeatType, typename LabelType>
		void CW_RDA<FeatType, LabelType>::BeginTrain() {
			Optimizer<FeatType, LabelType>::BeginTrain();
			this->u_t.zeros();
			this->sigma_w.set_value(1);
			this->gravity = 0;
		}

		//called when a train ends
		template <typename FeatType, typename LabelType>
		void CW_RDA<FeatType, LabelType>::EndTrain() {
			Optimizer<FeatType,LabelType>::EndTrain();
		}

		//set parameters for specific optimizers
		template <typename FeatType, typename LabelType>
		void CW_RDA<FeatType, LabelType>::SetParameterEx(float r) {
			this->r = r > 0 ? r : this->r;
		}

		//Change the dimension of weights
		template <typename FeatType, typename LabelType>
		void CW_RDA<FeatType, LabelType>::UpdateWeightSize(IndexType newDim) {
			if (newDim < this->weightDim)
				return;
			else {
				this->sigma_w.reserve(newDim + 1);
				this->sigma_w.resize(newDim + 1);
				//set the rest to one
				this->sigma_w.set_value(this->sigma_w.begin + this->weightDim, 
					this->sigma_w.end,1);

				this->u_t.reserve(newDim + 1);
				this->u_t.resize(newDim + 1);
				//set the rest to zero
				this->u_t.zeros(this->u_t.begin + this->weightDim,
					this->u_t.end);
				Optimizer<FeatType,LabelType>::UpdateWeightSize(newDim);
			}
		}
}

#endif
