/*************************************************************************
> File Name: STG.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/18 Sunday 17:25:54
> Functions: Sparse Online Learning With Truncated Gradient
> Reference:
Langford J, Li L, Zhang T. Sparse online learning via truncated 
gradient[J]. The Journal of Machine Learning Research, 2009, 10: 
777-801. 
************************************************************************/

#ifndef HEADER_STG
#define HEADER_STG

#include "SparseOnlineLinearModel.h"

/**
 *  namespace: Batch and Online Classification
 */
namespace BOC {
	template <typename FeatType, typename LabelType>
    class STG: public SparseOnlineLinearModel<FeatType, LabelType> {
		DECLARE_CLASS
    protected:
        int K;
        s_array<size_t> timeStamp;
        float (*pEta_time)(size_t t, float pt);

    public:
        STG(LossFunction<FeatType, LabelType> *lossFunc) :
            SparseOnlineLinearModel<FeatType, LabelType>(lossFunc){
                this->id_str = "STG";
                this->K = 10;
                this->timeStamp.resize(this->weightDim);
            }

        virtual ~STG(){
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
            printf("\tk	: %d\n",this->K);
        }
        /**
         * @Synopsis BeginTrain Reset the optimizer to the initialization status of training
         */
        virtual void BeginTrain() {
            SparseOnlineLinearModel<FeatType, LabelType>::BeginTrain();

            //reset time stamp
            this->timeStamp.zeros();

            if (this->power_t == 0.5)
                this->pEta_time = pEta_sqrt;
            else if(this->power_t == 0)
                this->pEta_time = pEta_const;
            else if (this->power_t == 1)
                this->pEta_time = pEta_linear;
            else
                this->pEta_time = pEta_general;
        }

        /**
         * @Synopsis EndTrain called when a train ends
         */
        virtual void EndTrain() {
            for (IndexType index_i = 1; index_i < this->weightDim; index_i++) {
                //truncated gradient
                size_t stepK = this->curIterNum - this->timeStamp[index_i];
                stepK -= stepK % this->K;

                this->weightVec[index_i] = trunc_weight(this->weightVec[index_i],
                        stepK * this->lambda * this->eta);
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
            this->curIterNum++;
            this->eta = this->eta0 / this->pEta_time(this->curIterNum, this->power_t);

            size_t featDim = x.indexes.size();
            float alpha = this->eta * this->lambda;

            float y = this->Predict(x); 
            float gt_i = this->lossFunc->GetGradient(x.label,y) * this->eta;

            size_t stepK = 0;
            for (size_t i = 0; i < featDim; i++) {
                IndexType index_i = x.indexes[i];
                //update the weight
                this->weightVec[index_i] -= gt_i * x.features[i];

                //lazy update the weight
                //truncated gradient
                if (this->timeStamp[index_i] == 0) {
                    this->timeStamp[index_i] = this->curIterNum;
                    continue;
                }
                else{
                    stepK = this->curIterNum - this->timeStamp[index_i];
                    if (stepK < size_t(this->K))
                        continue;

                    stepK -= stepK % this->K;
                    this->timeStamp[index_i] += stepK;
                    this->weightVec[index_i] = trunc_weight(this->weightVec[index_i],
                            stepK * alpha);
                }
            }
            //bias term
            this->weightVec[0] -= gt_i;
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

                SparseOnlineLinearModel<FeatType, LabelType>::UpdateModelDimention(new_dim);
            }
        }
    };
	IMPLEMENT_MODEL_CLASS(STG)
}
#endif
