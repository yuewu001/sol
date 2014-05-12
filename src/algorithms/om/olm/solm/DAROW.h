/*************************************************************************
  > File Name: Diagonal AROW
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: 2013/8/18 Sunday 17:25:54
  > Functions: Diagonal Adaptive Regularization of Weight Vectors
  > Reference: 
  Crammer, Koby, Alex Kulesza, and Mark Dredze. "Adaptive regularization 
  of weight vectors." Machine Learning (2009): 1-33.
 ************************************************************************/

#ifndef HEADER_D_AROW
#define HEADER_D_AROW

#include "SparseOnlineLinearModel.h"

/**
 *  namespace: Batch and Online Classification
 */
namespace BOC {
    template <typename FeatType, typename LabelType>
    class DAROW : public SparseOnlineLinearModel<FeatType, LabelType> {
		DECLARE_CLASS
        protected:
            float r;
            s_array<float> sigma_w;
            s_array<size_t> timeStamp;

        public:
            DAROW(LossFunction<FeatType, LabelType> *lossFunc) :
                SparseOnlineLinearModel<FeatType, LabelType>(lossFunc){
                    this->id_str = "DAROW";
                    this->r = 0;
                    this->sigma_w.resize(this->weightDim);
                    this->timeStamp.resize(this->weightDim);
                }
            virtual ~DAROW(){
            }

            /**
             * @Synopsis inherited functions
             */
        public:
            /**
             * PrintOptInfo print the info of optimization algorithm
             */
            virtual void PrintOptInfo() const {
                SparseOnlineLinearModel<FeatType, LabelType>::PrintOptInfo();
                printf("\tr:\t%g\n", this->r);
            }

            /**
             * @Synopsis SetParameter set parameters for the learning model
             *
             * @Param param
             */
            virtual void SetParameter(BOC::Params &param){
                OnlineLinearModel<FeatType, LabelType>::SetParameter(param);
                this->r = param.FloatValue("-r");
            }

            /**
             * @Synopsis BeginTrain Reset the optimizer to the initialization status of training
             */
            virtual void BeginTrain() {
                SparseOnlineLinearModel<FeatType, LabelType>::BeginTrain();

                this->timeStamp.zeros();
                this->sigma_w.set_value(1);
            }

            /**
             * @Synopsis EndTrain called when a train ends
             */
            virtual void EndTrain() {
                float beta_t = 1.f / this->r;
                float temp_beta = beta_t * this->lambda /2.f;

                for (IndexType index_i = 1; index_i < this->weightDim; index_i++) {
                    //L1 lazy update
                    size_t stepK = this->curIterNum - this->timeStamp[index_i];
                    if (stepK == 0)
                        continue;
                    this->timeStamp[index_i] = this->curIterNum;

                    this->weightVec[index_i] = trunc_weight(this->weightVec[index_i],
                            stepK * temp_beta);
                }
                SparseOnlineLinearModel<FeatType, LabelType>::EndTrain();
            }

            /**
             * @Synopsis Iterate Iteration of online learning
             *
             * @Param x current input data example
             *
             * @Returns  prediction of the current example
             */
            virtual float Iterate(const DataPoint<FeatType, LabelType> &x) {
                size_t featDim = x.indexes.size();
                float y = this->Predict(x);
                //calculate beta_t
                float beta_t = this->r;
                for (size_t i = 0; i < featDim; i++){
                    beta_t += x.features[i] * x.features[i] * this->sigma_w[x.indexes[i]];
                }
                beta_t = 1.f / beta_t;
                float temp_beta = beta_t * this->lambda / 2.f;

                IndexType index_i = 0;
                //y /= this->curIterNum;
                //calculate beta_t

                float alpha_t = 1 - x.label * y;
                if(alpha_t > 0){
                    alpha_t *= beta_t; 
                    for (size_t i = 0; i < featDim; i++){
                        index_i = x.indexes[i];
                        //update u_t
                        this->weightVec[index_i] += alpha_t * 
                            this->sigma_w[index_i] * x.label * x.features[i];

                        //L1 lazy update
                        size_t stepK = this->curIterNum - this->timeStamp[index_i];
                        this->timeStamp[index_i] = this->curIterNum;

                        this->weightVec[index_i]= 
                            trunc_weight(this->weightVec[index_i],stepK * temp_beta);
                        //update sigma_w
                        this->sigma_w[index_i] *= this->r / (this->r + 
                                this->sigma_w[index_i] * x.features[i] * x.features[i]);
                        /*
                           this->sigma_w[index_i] -= beta_t * 
                           this->sigma_w[index_i] * this->sigma_w[index_i] * 
                           x.features[i] * x.features[i];
                           */
                    }

                    //bias term
                    this->weightVec[0] += alpha_t * this->sigma_w[0] * x.label;
                    //this->sigma_w[0] -= beta_t * this->sigma_w[0] * this->sigma_w[0];
                    this->sigma_w[0] *= this->r / (this->r + this->sigma_w[0]);
                }
                return y;
            }

            /**
             * @Synopsis UpdateModelDimention update dimension of the model,
             * often caused by the increased dimension of data
             *
             * @Param new_dim new dimension
             */
            virtual void UpdateModelDimention(IndexType new_dim) {
                if (new_dim < this->weightDim)
                    return;
                else {
                    this->timeStamp.reserve(new_dim + 1);
                    this->timeStamp.resize(new_dim + 1);
                    //set the rest to zero
                    this->timeStamp.zeros(this->timeStamp.begin + this->weightDim,
                            this->timeStamp.end);

                    this->sigma_w.reserve(new_dim + 1);
                    this->sigma_w.resize(new_dim + 1);  //reserve the 0-th
                    //set the rest to 1
                    this->sigma_w.set_value(this->sigma_w.begin + this->weightDim,
                            this->sigma_w.end, 1);

                    SparseOnlineLinearModel<FeatType, LabelType>::UpdateModelDimention(new_dim);
                }
            }

    };
	IMPLEMENT_CLASS(DAROW)
}

#endif
