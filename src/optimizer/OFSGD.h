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
#include <algorithm>
#include <math.h>
#include <queue>

namespace SOL {
	struct IndexValuePair{
		IndexType index;
		float weight;
		IndexValuePair(IndexType i, float w):index(i), weight(w){}

		friend bool operator < (const IndexValuePair &one, const IndexValuePair & other){
			return one.weight > other.weight;
		}
	};

    template <typename FeatType, typename LabelType>
        class OFSGD: public Optimizer<FeatType, LabelType> {
            protected:
                IndexType K; //keep top K elemetns

                float w_norm;
                float norm_coeff;

				float shrinkage;
				
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
                    printf("eta:\t%.2f\n", this->eta0);
                    printf("lambda:\t%.2f\n", this->lambda);
                    printf("K:\t%d\n\n", this->K);
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
				float y = this->Predict(x);
				size_t featDim = x.indexes.size();
				this->eta = this->eta0 / this->pEta_time(this->curIterNum, this->power_t);

				//first shrinkage
				this->shrinkage = 1.f - this->eta *  this->lambda;
				for (IndexType i = 0; i < this->weightDim; i++) {
					this->weightVec[i] *= this->shrinkage;
				}
				w_norm *= (this->shrinkage * this->shrinkage);

				float gt_i = this->lossFunc->GetGradient(x.label,y);
				if (gt_i == 0){
					return y;
				}

				IndexType index_i = 0;
				for (size_t i = 0; i < featDim; i++) {
					index_i = x.indexes[i];
					w_norm -= this->weightVec[index_i] * this->weightVec[index_i];
					this->weightVec[index_i] -= this->eta * gt_i * x.features[i];
					w_norm += this->weightVec[index_i] * this->weightVec[index_i];
				}
				//update bias 
				w_norm -= this->weightVec[0] * this->weightVec[0];
				this->weightVec[0] -= this->eta * gt_i;
				w_norm += this->weightVec[0] * this->weightVec[0];

				//normlize
				float coeff = this->norm_coeff / sqrtf(w_norm);
				if (coeff < 1){
					for (IndexType i = 0; i < this->weightDim; i++)
						this->weightVec[i] *= coeff;
					w_norm *= (coeff * coeff);
				}

				//truncate
				std::priority_queue<IndexValuePair> truncator;
				for (IndexType i = 0; i < this->K; i++){
					truncator.push(IndexValuePair(i,(std::abs)(this->weightVec[i])));
				}
				IndexValuePair pair = truncator.top();
				for (IndexType i = K; i < this->weightDim; i++){
					float curWeight = (std::abs)(this->weightVec[i]);
					if (pair.weight >= curWeight)
						continue;
					else{
						truncator.pop();
						this->weightVec[pair.index] = 0;
						truncator.push(IndexValuePair(i,curWeight));
						pair = truncator.top();
					}
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
			this->w_norm = 0;
			this->norm_coeff = 1.f / sqrtf(this->lambda);

			this->shrinkage = 1.f - this->lambda * this->eta0;
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
			}
		}
}

#endif
