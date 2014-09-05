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

#include "OnlineLinearModel.h"

/**
 *  namespace: Batch and Online Classification
 */
namespace BOC {
	template <typename FeatType, typename LabelType>
	class DAROW : public OnlineLinearModel < FeatType, LabelType > {

		DECLARE_CLASS

	protected:
		float r;
		vector<s_array<float> > sigmaWMatrix;

		//accepted loss functions
		vector<string> bc_loss_funcs;
		vector<string> mc_loss_funcs;
	public:
		DAROW(LossFunction<FeatType, LabelType> *lossFunc, int classNum) :
			OnlineLinearModel<FeatType, LabelType>(lossFunc, classNum){
			this->modelName = "DAROW";
			this->r = 0;

			this->sigmaWMatrix.resize(this->classfier_num);
			for (int k = 0; k < this->classfier_num; ++k){
				this->sigmaWMatrix[k].resize(this->weightDim);
			}

			this->bc_loss_funcs.push_back(SquaredHingeLoss<FeatType, LabelType>::GetClassMsg().GetType());
			this->mc_loss_funcs.push_back(MaxScoreSquaredHingeLoss<FeatType, LabelType>::GetClassMsg().GetType());
			this->mc_loss_funcs.push_back(UniformSquaredHingeLoss<FeatType, LabelType>::GetClassMsg().GetType());

			//set up the loss function
			const std::string& loss_type = this->lossFunc->GetType();
			vector<string> * loss_funcs = NULL;
			if (this->classfier_num == 1){
				loss_funcs = &this->bc_loss_funcs;
			}
			else{
				loss_funcs = &this->mc_loss_funcs;
			}

			bool is_supported = false;
			for (vector<string>::iterator iter = loss_funcs->begin(); iter != loss_funcs->end(); ++iter){
				if (*iter == loss_type){
					is_supported = true;
					break;
				}
			}
			if (is_supported == false){
				std::ostringstream oss;
				oss << "specified loss function " << loss_type << " is not supported by " << this->GetType() << ". Please check the document for details!";
				throw invalid_argument(oss.str());
			}
		}

		virtual ~DAROW(){
		}

		/**
		 * @Synopsis inherited functions
		 */
	public:
		/**
		 * PrintModelSettings print the info of optimization algorithm
		 */
		virtual void PrintModelSettings() const {
			OnlineLinearModel<FeatType, LabelType>::PrintModelSettings();
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
			INVALID_ARGUMENT_EXCEPTION(r, this->r >= 0, "no smaller than 0");
		}

		/**
		 * @Synopsis BeginTrain Reset the optimizer to the initialization status of training
		 */
		virtual void BeginTrain() {
			OnlineLinearModel<FeatType, LabelType>::BeginTrain();

			for (int i = 0; i < this->classfier_num; ++i){
				this->sigmaWMatrix[i].set_value(1);
			}
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

				for (int i = 0; i < this->classfier_num; ++i){
					s_array<float>& sigma_w = this->sigmaWMatrix[i];

					sigma_w.resize(new_dim + 1);
					sigma_w.set_value(sigma_w.begin + this->weightDim, sigma_w.end, 1);
				}

				OnlineLinearModel<FeatType, LabelType>::UpdateModelDimention(new_dim);
			}
		}

	protected:
		/**
		 * @Synopsis UpdateWeightVec Update the weight vector
		 *
		 * @Param x current input data example
		 * @Param gt common part of the gradient
		 *
		 */
		virtual void UpdateWeightVec(const DataPoint<FeatType, LabelType> &x, float* gt_t){
			size_t featDim = x.indexes.size();
			IndexType index_i = 0;
			//calculate beta_t
			float beta_t = this->r;
			for (int k = 0; k < this->classfier_num; ++k){
				if (gt_t[k] == 0){
					continue;
				}
				//classifier weight
				float cw = this->classifier_weight[k] * this->classifier_weight[k];

				s_array<float>& sigma_w = this->sigmaWMatrix[k];
				for (size_t i = 0; i < featDim; i++){
					beta_t += cw * x.features[i] * x.features[i] * sigma_w[x.indexes[i]];
				}
			}

			beta_t = 0.5f / beta_t;

			for (int k = 0; k < this->classfier_num; ++k){
				if (gt_t[k] == 0){
					continue;
				}

				//classifier weight
				float cw = this->classifier_weight[k] * this->classifier_weight[k];

				s_array<float>& weightVec = this->weightMatrix[k];
				s_array<float>& sigma_w = this->sigmaWMatrix[k];
				for (size_t i = 0; i < featDim; ++i){
					index_i = x.indexes[i];
					//update u_t
					weightVec[index_i] -= beta_t * sigma_w[index_i] * gt_t[k] * x.features[i];


					//update sigma_w
					sigma_w[index_i] *= this->r / (this->r +
						sigma_w[index_i] * x.features[i] * x.features[i] * cw);
				}

				//bias term
				weightVec[0] -= beta_t * sigma_w[0] * gt_t[k];
				//this->sigma_w[0] -= beta_t * this->sigma_w[0] * this->sigma_w[0];
				sigma_w[0] *= this->r / (this->r + sigma_w[0] * cw);
			}
		}
	};

	IMPLEMENT_MODEL_CLASS(DAROW, "Diagonal AROW")
}

#endif
