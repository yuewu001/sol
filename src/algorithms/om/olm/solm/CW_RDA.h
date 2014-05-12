/*************************************************************************
> File Name: CW_RDA.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/11/27 14:04:06
> Functions: Confidence weighted regularized dual averaging
************************************************************************/
#ifndef HEADER_CW_RDA
#define HEADER_CW_RDA

#include "SparseOnlineLinearModel.h"

/**
*  namespace: Batch and Online Classification
*/
namespace BOC {
	template <typename FeatType, typename LabelType>
	class CW_RDA: public SparseOnlineLinearModel<FeatType, LabelType> {
		DECLARE_CLASS
	protected:
		float r;
		s_array<float> sigma_w;
		s_array<float> u_t;
		float gravity;

	public:
		CW_RDA(LossFunction<FeatType, LabelType> *lossFunc) :
			SparseOnlineLinearModel<FeatType, LabelType>(lossFunc){
                this->id_str = "Confidence Weighted RDA";
                this->r = 0;
                this->u_t.resize(this->weightDim);
                this->sigma_w.resize(this->weightDim);
                this->lossFunc = new SquaredHingeLoss<FeatType, LabelType>;
            }

        virtual ~CW_RDA(){
            if (this->lossFunc != NULL){
                delete this->lossFunc;
                this->lossFunc = NULL;
            }
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
            this->u_t.zeros();
            this->sigma_w.set_value(1);
            this->gravity = 0;
        }

        /**
		 * @Synopsis Iterate Iteration of online learning
		 *
		 * @Param x current input data example
		 *
		 * @Returns  prediction of the current example
		 */
		virtual float Iterate(const DataPoint<FeatType, LabelType> &x) {
            IndexType* p_index = x.indexes.begin;
            float* p_feat = x.features.begin;
            //obtain w_t
            while(p_index != x.indexes.end){
                //lazy update
                //this->weightVec[*p_index] = -sqrtf(this->sigma_w[*p_index]) *
                this->weightVec[*p_index] = -sqrtf(this->sigma_w[*p_index]) *
                    trunc_weight(u_t[*p_index], gravity);
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
                gravity += this->lambda * this->eta;
                gt *= this->eta;
                //update
                p_index = x.indexes.begin;
                p_feat = x.features.begin;
                while(p_index != x.indexes.end){
                    this->u_t[*p_index++] += gt * (*p_feat++);
                }

                //bias term
                this->u_t[0] += gt;
                this->weightVec[0] = -u_t[0] * sqrtf(this->sigma_w[0]); 
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
                this->sigma_w.reserve(new_dim + 1);
                this->sigma_w.resize(new_dim + 1);
                //set the rest to one
                this->sigma_w.set_value(this->sigma_w.begin + this->weightDim, 
                        this->sigma_w.end,1);

                this->u_t.reserve(new_dim + 1);
                this->u_t.resize(new_dim + 1);
                //set the rest to zero
                this->u_t.zeros(this->u_t.begin + this->weightDim,
                        this->u_t.end);

                SparseOnlineLinearModel<FeatType, LabelType>::UpdateModelDimention(new_dim);
            }
        }
    };
	IMPLEMENT_CLASS(CW_RDA)
}

#endif
