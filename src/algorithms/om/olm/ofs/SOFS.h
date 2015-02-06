/*************************************************************************
> File Name: Sparse Diagonal AROW
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/18 Sunday 17:25:54
> Functions: second order online feature selection
************************************************************************/

#ifndef HEADER_SOFS
#define HEADER_SOFS

#include "OnlineFeatureSelection.h"
#include "../../../../utils/MaxHeap.h"

#include <sstream>

/**
*  namespace: Batch and Online Classification
*/
namespace BOC {
	template <typename FeatType, typename LabelType>
	class SOFS : public OnlineFeatureSelection < FeatType, LabelType > {

		DECLARE_CLASS

#pragma region Class Members
	protected:
		float r;
		vector<s_array<float> > sigmaWMatrix;
		s_array<float> sigmaWSum;
		MaxHeap<float> heap;

        //accepted loss functions
		vector<string> bc_loss_funcs;
		vector<string> mc_loss_funcs;

#pragma endregion Class Members


#pragma region Constructors and Basic Functions
	public:
		SOFS(LossFunction<FeatType, LabelType> *lossFunc, int classNum) :
			OnlineFeatureSelection<FeatType, LabelType>(lossFunc, classNum) {
			this->modelName = "SOFS";
			this->r = init_r;

			this->sigmaWMatrix.resize(this->classfier_num);
			for (int k = 0; k < this->classfier_num; ++k){
				this->sigmaWMatrix[k].resize(this->weightDim);
			}

			this->sigmaWSum.resize(this->weightDim);

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

		virtual ~SOFS(){
		}

		/**
		 * @Synopsis inherited functions
		 */
	public:
		/**
		 * PrintModelSettings print the info of optimization algorithm
		 */
		virtual void PrintModelSettings() const {
			OnlineFeatureSelection<FeatType, LabelType>::PrintModelSettings();
			printf("\tr:\t%g\n", this->r);
		}

#pragma region Constructors and Basic Functions

#pragma region Train Related
	public:
		/**
		 * @Synopsis SetParameter set parameters for the learning model
		 *
		 * @Param param
		 */
		virtual void SetParameter(BOC::Params &param) {
			OnlineFeatureSelection<FeatType, LabelType>::SetParameter(param);
			float val = param.FloatValue("-r");
			if (val >= 0){
				this->r = val;
			}
		}

		/**
		 * @Synopsis BeginTrain Reset the optimizer to the initialization status of training
		 */
		virtual void BeginTrain() {
			INVALID_ARGUMENT_EXCEPTION(r, this->r >= 0, "no smaller than 0");
			OnlineFeatureSelection<FeatType, LabelType>::BeginTrain();

			if (this->weightDim < this->K + 1){
				this->UpdateModelDimention(this->K); //remove the bais term
			}

			for (int i = 0; i < this->classfier_num; ++i){
				this->sigmaWMatrix[i].set_value(1);
			}

			this->sigmaWSum.set_value((float)(this->classfier_num));

			heap.Init(this->weightDim - 1, this->K, this->sigmaWSum.begin + 1);
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

				this->sigmaWSum.resize(new_dim + 1);  //reserve the 0-th
				//set the rest to 1
				this->sigmaWSum.set_value(this->sigmaWSum.begin + this->weightDim,
					this->sigmaWSum.end, (float)this->classfier_num);
				heap.UpdateDataNum(new_dim, this->sigmaWSum.begin + 1);

				OnlineFeatureSelection<FeatType, LabelType>::UpdateModelDimention(new_dim);
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
					/*
					if (this->heap.is_topK(index_i - 1)){
						//update u_t
						weightVec[index_i] -= beta_t * sigma_w[index_i] * gt_t[k] * x.features[i];
					}
					else{
						weightVec[index_i] = 0;
					}
					*/

					//update sigma_w
					this->sigmaWSum[index_i] -= sigma_w[index_i];
					sigma_w[index_i] *= this->r / (this->r +
						sigma_w[index_i] * x.features[i] * x.features[i] * cw);
					this->sigmaWSum[index_i] += sigma_w[index_i];
				}

				//bias term
				weightVec[0] -= beta_t * sigma_w[0] * gt_t[k];
				//this->sigma_w[0] -= beta_t * this->sigma_w[0] * this->sigma_w[0];
				sigma_w[0] *= this->r / (this->r + sigma_w[0] * cw);
			}

			//update the heap
			for (size_t i = 0; i < featDim; i++){
				IndexType ret_id;
				if (this->heap.UpdateHeap(x.indexes[i] - 1, ret_id) == true){
					++ret_id;
					for (int k = 0; k < this->classfier_num; ++k){
						this->weightMatrix[k][ret_id] = 0;
					}
				}
			}
		}
#pragma endregion Train Related

	};

	IMPLEMENT_MODEL_CLASS(SOFS, "Second Order Online Feature Selection")
}
#endif
