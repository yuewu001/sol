/*************************************************************************
	> File Name: SparseOnlineLinearModel.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 5/5/2014 4:54:43 PM
	> Functions: interfaces for sparse linear online model
 ************************************************************************/
#ifndef HEADER_SPARSE_LINEAR_ONLINE_MODEL
#define HEADER_SPARSE_LINEAR_ONLINE_MODEL

#include "../OnlineLinearModel.h"

/**
*  namespace: Batch and Online Classification
*/
namespace BOC {
	template <typename FeatType, typename LabelType> 
    class SparseOnlineLinearModel : public OnlineLinearModel<FeatType, LabelType> {
        protected:
            //L1 regularization parameter
            float lambda;
            //weights below this threshold will eliminated at the end of training
            float sparse_soft_thresh;

        public:
            SparseOnlineLinearModel(LossFunction<FeatType, LabelType> &lossFunc) : 
                OnlineLinearModel<FeatType, LabelType>(lossFunc) {
                    this->lambda = 0;
                    this->sparse_soft_thresh = init_sparse_soft_thresh;
                }

            virtual ~SparseOnlineLinearModel() {
            }

            //inherited functions
        public:
            /**
             * PrintOptInfo print the info of optimization algorithm
             */
            virtual void PrintOptInfo() const {
                OnlineLinearModel<FeatType, LabelType>::PrintOptInfo();

                printf("Sparse Online Linear Model\n");
                printf("\tl1 regularization: %g\n", this->lambda);
            }

            /**
             * @Synopsis SetParameter set parameters for the learning model
             *
             * @Param param
             */
			virtual void SetParameter(BOC::Params &param){
                OnlineLinearModel<FeatType, LabelType>::SetParameter(param);
				this->lambda = param.FloatValue("-l1");
            }

            /**
             * @Synopsis EndTrain called when a train ends
             */
            virtual void EndTrain() {
                //eliminate weights smaller than sparse_soft_thresh
				for (IndexType i = 1; i < this->weightDim; i++){
					if (this->weightVec[i] < this->sparse_soft_thresh &&
						this->weightVec[i] > -this->sparse_soft_thresh){
						this->weightVec[i] = 0;
					}
				}

				OnlineLinearModel<FeatType, LabelType>::EndTrain();
			}

	protected:
		/**
		 * @Synopsis SaveModelConfig save configuration of model to disk
		 *
		 * @Param os ostream object to which config are saved
		 *
		 * @Returns true if saved successfully
		 */
		virtual bool SaveModelConfig(std::ofstream &os) {
			OnlineLinearModel<FeatType, LabelType>::SaveModelConfig(os);

			//l1 regularization
			os << "lambda: " << this->lambda << "\n";

			return true;
		}

		/**
		 * @Synopsis LoadModelConfig load configuration of model from disk
		 *
		 * @Param is istream object from which config are loaded
		 *
		 * @Returns true if load successfully
		 */
		virtual bool LoadModelConfig(std::ifstream &is) {
			OnlineLinearModel<FeatType, LabelType>::LoadModelConfig(is);

			string line;
			//l1 regularization
			getline(is, line, ':');
			is >> this->lambda;
			getline(is, line);

			return true;
		}

		/**
		 * @Synopsis newly defined functions
		 */
	public:
		/**
		 * @Synopsis GetSparseRate get the sparse rate of linear model
		 *
		 * @Param total_len users specified dimension of weights
		 *
		 * @Returns sparse rate
		 */
		float GetSparseRate(IndexType total_len = 0) {
			if (this->weightDim == 1)
				return 1;
			IndexType zeroNum = this->weightDim - 1 - this->GetNonZeroNum();

			if (total_len > 0)
				return zeroNum / (float)total_len;
			else
				return zeroNum / (float)(this->weightDim - 1);
		}

		/**
		 * @Synopsis GetNonZeroNum get the number of nonzero weights
		 *
		 * @Returns number of nonzero weights
		 */
		IndexType GetNonZeroNum() {
			IndexType non_zeroNum(0);
			if (this->weightDim == 1)
				return 0;

			for (IndexType i = 1; i < this->weightDim; i++) {
				if (this->weightVec[i] != 0)
					non_zeroNum++;
			}
			return non_zeroNum;
		}
	};

}

#endif

