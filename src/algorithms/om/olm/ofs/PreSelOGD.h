/************************************************************************
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: 8/19/2013 Monday 10:37:08 AM
  > Functions: Stochastic Gradient Descent with pre-selected features
  ************************************************************************/

#ifndef HEADER_PRE_SELECTED_OGD
#define HEADER_PRE_SELECTED_OGD

#include "OnlineFeatureSelection.h"
#include "../../../../io/basic_io.h"

#include <string>

namespace BOC {
	template <typename FeatType, typename LabelType>
	class PreSelOGD : public OnlineFeatureSelection < FeatType, LabelType > {

		DECLARE_CLASS

#pragma region Class Members
	protected:
		s_array<char> sel_feat_flag_vec;
		IndexType max_index;

		float(*pEta_time)(size_t t, float pt);
#pragma endregion Class Members

#pragma region Constructors and Basic Functions
	public:
		PreSelOGD(LossFunction<FeatType, LabelType> *lossFunc, int classNum) :
			OnlineFeatureSelection<FeatType, LabelType>(lossFunc, classNum) {
			this->modelName = "PreSelOGD";
			this->max_index = 0;
		}

#pragma endregion Constructors and Basic Functions

#pragma region Train Related
	public:
		/**
		 * @Synopsis SetParameter set parameters for the learning model
		 *
		 * @Param param
		 */
		virtual void SetParameter(BOC::Params &param) {
			OnlineFeatureSelection<FeatType, LabelType>::SetParameter(param);
			const std::string& modelFileName = param.StringValue("-im");

			if (this->LoadFSResult(modelFileName) != STATUS_OK){
				throw invalid_argument("load input model file failed!");
			}
		}

		/**
		 * @Synopsis BeginTrain Reset the optimizer to the initialization status of training
		 */
		virtual void BeginTrain() {
			OnlineFeatureSelection<FeatType, LabelType>::BeginTrain();

			if (this->power_t == 0.5)
				this->pEta_time = pEta_sqrt;
			else if (this->power_t == 0)
				this->pEta_time = pEta_const;
			else if (this->power_t == 1)
				this->pEta_time = pEta_linear;
			else
				this->pEta_time = pEta_general;
		}


	protected:
		/**
		 * @Synopsis UpdateWeightVec Update the weight vector
		 *
		 * @Param x current input data example
		 * @Param weightVec weight vector to be updated
		 * @param gt common part of the gradient
		 * @Param beta extra multiplier for updating, if none, set it to 1
		 *
		 */
		virtual void UpdateWeightVec(const DataPoint<FeatType, LabelType> &x, float* gt_t){
			this->eta = this->eta0 / this->pEta_time(this->curIterNum, this->power_t);
			size_t featDim = x.indexes.size();

			//update with sgd
			for (int k = 0; k < this->classfier_num; ++k){
				s_array<float> &weightVec = this->weightMatrix[k];
				for (size_t i = 0; i < featDim; i++) {
					if (x.indexes[i] <= this->max_index &&
						this->sel_feat_flag_vec[x.indexes[i]] == 1)
						weightVec[x.indexes[i]] -= this->eta * gt_t[k] * x.features[i];

				}
				//update bias 
				weightVec[0] -= this->eta * gt_t[k];
			}
		}


	protected:
		//load the feature selection result
		int LoadFSResult(const string& filename, int k = -1){
			this->max_index = 0;
			this->sel_feat_flag_vec.erase();

			basic_io io_handler;
			io_handler.open_file(filename.c_str(), "r");
			int err_code = io_handler.good();
			if (err_code != 0){
				fprintf(stderr, "open file %s failed, error code: %d\n!", filename.c_str(), err_code);
				return err_code;
			}

			IndexType featIndex = 0;
			vector<IndexType> featVec;

			//load feature indexes
			size_t dst_len = 64;
			char* line = new char[dst_len];
			while (io_handler.read_line(line, dst_len) != NULL){
				if (line[0] == '\n' || line[0] == '\r')
					continue;
				featIndex = (IndexType)(atoi(line));
				if (featIndex == 0){
					fprintf(stderr, "parse index %s failed!\n", line);
					err_code = 1;
					break;
				}
				featVec.push_back(featIndex);

				if (k > 0){
					if (--k == 0)break;
				}
			}
			delete[]line;

			//find the max index
			this->max_index = *std::max_element(featVec.begin(), featVec.end());
			this->sel_feat_flag_vec.resize(this->max_index + 1);
			this->sel_feat_flag_vec.zeros();

			for (vector<IndexType>::iterator iter = featVec.begin(); iter != featVec.end(); ++iter){
				this->sel_feat_flag_vec[*iter] = 1;
			}

			this->K = (int)(featVec.size());
			return err_code;
		}
#pragma endregion Train Related
	};

	IMPLEMENT_MODEL_CLASS(PreSelOGD, "Pre-Selected Online Gradient Descent")

}
#endif
