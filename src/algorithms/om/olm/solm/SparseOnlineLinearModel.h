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

		s_array<float>& weightVec;

	public:
		SparseOnlineLinearModel(LossFunction<FeatType, LabelType> *lossFunc) 
			: OnlineLinearModel<FeatType, LabelType>(lossFunc) ,
			weightVec(*this->pWeightVecBC){
				this->lambda = 0;
				this->sparse_soft_thresh = init_sparse_soft_thresh;
			}

		virtual ~SparseOnlineLinearModel() {
		}

		//inherited functions
	public:
		/**
		 * PrintModelSettings print the info of optimization algorithm
		 */
		virtual void PrintModelSettings() const {
			OnlineLinearModel<FeatType, LabelType>::PrintModelSettings();

			printf("Linear Sparse Online Learning:\n");
			printf("\tl1 regularization: %g\n", this->lambda);
		}

		/**
		 * PrintModelInfo print the info of trained model
		 */
		virtual void PrintModelInfo() const {
			OnlineLinearModel<FeatType, LabelType>::PrintModelInfo();

			IndexType nonZeroNum = this->GetNonZeroNum();
			printf("Non-Zero weight number: %u\n", nonZeroNum);

			double sparseRate = 0;
			if (this->weightDim == 1){
				sparseRate = 0;
			}
			else{
				sparseRate = (this->weightDim - 1 - nonZeroNum) / (double)(this->weightDim - 1);
			}

			printf("Sparsification Rate: %g %%\n", sparseRate * 100);
		}

		/**
		 * @Synopsis SetParameter set parameters for the learning model
		 *
		 * @Param param
		 */
		virtual void SetParameter(BOC::Params &param){
			OnlineLinearModel<FeatType, LabelType>::SetParameter(param);
			this->lambda = param.FloatValue("-l1");
			INVALID_ARGUMENT_EXCEPTION(lambda, this->lambda >= 0, "no smaller than 0");
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



		/**
		 * @Synopsis IterateBC Iteration of online learning for binary classification
		 *
		 * @Param x current input data example
		 *
		 * @Returns  predicted class of the current example
		 */
		virtual int IterateBC(const DataPoint<FeatType, LabelType> &x, float& predict){
			predict = this->Iterate(x);
			int label = this->GetClassLabel(x);
			if (this->IsCorrect(label, predict) == false){
				return -label;
			}
			else{
				return x.label;
			}
		}

		/**
		 * @Synopsis IterateMC Iteration of online learning for multiclass classification
		 *
		 * @Param x current input data example
		 *
		 * @Returns  predicted class of the current example
		 */
		virtual int IterateMC(const DataPoint<FeatType, LabelType> &x, float& predict){
			fprintf(stderr, "multiclass is not supported yet!");
			exit(1);
		}

	protected:
		/**
		 * @Synopsis Iterate Iteration of online learning
		 *
		 * @Param x current input data example
		 *
		 * @Returns  prediction of the current example
		 */
		virtual float Iterate(const DataPoint<FeatType, LabelType> &x) = 0;

		/**
		 * @Synopsis UpdateWeightVec Update the weight vector
		 *
		 * @Param x current input data example
		 * @Param weightVec weight vector to be updated
		 * @param gt common part of the gradient
		 */
		virtual void UpdateWeightVec(const DataPoint<FeatType, LabelType> &x, s_array<float>& weightVec, float gt){}

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

	public:
		/**
		 * @Synopsis GetNonZeroNum get the number of nonzero weights
		 *
		 * @Returns number of nonzero weights
		 */
		IndexType GetNonZeroNum()  const {
			IndexType nonZeroNum = 0;
			for (IndexType i = 1; i < this->weightDim; ++i){
				if (this->weightVec[i] != 0){
					++nonZeroNum;
				}
			}
			return nonZeroNum;
		}
	};

}

#endif

