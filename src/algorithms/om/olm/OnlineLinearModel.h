/*************************************************************************
	> File Name: OnlineAlgorithm.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 5/5/2014 2:53:17 PM
	> Functions: interface definition of online linear model
	************************************************************************/
#ifndef HEADER_ONLINE_LINEAR_MODEL
#define HEADER_ONLINE_LINEAR_MODEL

#include "../OnlineModel.h"

#include <fstream>
#include <string>
#include <algorithm>

/**
*  namespace: Batch and Online Classification
*/
namespace BOC {
	template <typename FeatType, typename LabelType>
	class OnlineLinearModel : public OnlineModel < FeatType, LabelType > {

#pragma region Class Members
		//weight vector
	protected:
		//the first element is zero
		vector<s_array<float> > weightMatrix;
		//weight vector used for bc
		s_array<float> *pWeightVecBC;
		//gradients for multi-class
		vector<float> mc_gradients;

		//weight dimension: can be the same to feature, or with an extra bias
		IndexType weightDim;
#pragma endregion Class Members

#pragma region Constructors and Basic Functions
	public:
		OnlineLinearModel(LossFunction<FeatType, LabelType> *lossFunc, int classNum)
			: OnlineModel<FeatType, LabelType>(lossFunc, classNum), pWeightVecBC(NULL) {
			this->weightDim = 1;
			this->weightMatrix.resize(this->classfier_num);

			for (int i = 0; i < this->classfier_num; ++i){
				//weight vector
				this->weightMatrix[i].resize(this->weightDim);
			}
			if (this->classfier_num == 1){
				this->pWeightVecBC = &this->weightMatrix[0];
			}
			else{
				this->mc_gradients.resize(this->classfier_num);
			}
		}

		virtual ~OnlineLinearModel() {
		}

		/**
		 * PrintModelSettings print the info of optimization algorithm
		 */
		virtual void PrintModelSettings() const {
			OnlineModel<FeatType, LabelType>::PrintModelSettings();

			printf("Linear Model: y = w * x + b \n");
		}

		/**
		 * PrintModelInfo print the info of trained model
		 */
		virtual void PrintModelInfo() const {
			printf("number of weights: %lu\n", static_cast<long int>(this->weightDim));
		}
#pragma endregion Constructors and Basic Functions

#pragma region  IO related
	public:
		/**
		 * @Synopsis SaveModel save model to disk
		 *
		 * @Param filename  name to the saved file
		 *
		 * @Returns true if load successfully
		 */
		virtual bool SaveModel(const string& filename) {
			std::ofstream outfile(filename.c_str(), ios::out | ios::binary);
			if (!outfile){
				fprintf(stderr, "open file %s failed!\n", filename.c_str());
				return false;
			}

			this->SaveModelConfig(outfile);
			this->SaveModelValue(outfile);

			outfile.close();
			return true;
		}

		/**
		 * @Synopsis LoadModel load model from disk
		 *
		 * @Param filename  path name of the model on disk
		 *
		 * @Returns true if load successfully
		 */
		virtual bool LoadModel(const string& filename) {
			std::ifstream infile(filename.c_str(), ios::in | ios::binary);
			if (!infile){
				fprintf(stderr, "open file %s failed!\n", filename.c_str());
				return false;
			}

			this->LoadModelConfig(infile);
			this->LoadModelValue(infile);

			infile.close();
			return true;
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
			//model
			os << "[model]\n";
			os << "y = w * x + b\n";
			//current iteration step
			os << "current iteration number: " << this->curIterNum << "\n";
			//current learning rate
			os << "current learning rate: " << this->eta << "\n";
			//power_t for learning rate
			os << "power_t: " << this->power_t << "\n";

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
			//model
			std::string line;
			getline(is, line);
			getline(is, line);

			//current iteration step
			getline(is, line, ':');
			is >> this->curIterNum;
			getline(is, line);

			//current learning rate
			getline(is, line, ':');
			is >> this->eta;
			getline(is, line);

			//power_t for learning rate
			getline(is, line, ':');
			is >> this->power_t;
			getline(is, line);

			return true;
		}

		/**
		 * @Synopsis  SaveModelValue save model value to disk
		 *
		 * @Param os ostream object to which values are saved
		 *
		 * @Returns true if saved successfully
		 */
		virtual bool SaveModelValue(std::ofstream &os) {
			//weight dimension
			os << "[value]\n";
			os << "classfier num: " << this->classfier_num << "\n";
			os << "weight dimension: " << this->weightDim << "\n";
			//weights
			for (int k = 0; k < this->classfier_num; ++k){
				os << k << " | ";
				s_array<float>& weightVec = this->weightMatrix[k];
				for (IndexType i = 0; i < this->weightDim; i++){
					if (weightVec[i] != 0){
						os << i << ":" << weightVec[i] << "\t";
					}
				}
				os << "\n";
			}

			return true;
		}

		/**
		 * @Synopsis LoadModelConfig load values of model from disk
		 *
		 * @Param is istream object from which values are loaded
		 *
		 * @Returns true if load successfully
		 */
		virtual bool LoadModelValue(std::ifstream &is) {
			//weight dimension
			string line;

			getline(is, line);
			getline(is, line, ':');
			is >> this->classfier_num;
			getline(is, line);

			getline(is, line, ':');
			is >> this->weightDim;
			getline(is, line);

			this->weightMatrix.resize(this->classfier_num);
			for (int k = 0; k < this->classfier_num; ++k){
				s_array<float>& weightVec = this->weightMatrix[k];
				weightVec.resize(this->weightDim);
				weightVec.zeros();
				getline(is, line, '|');
				//weights
				getline(is, line);
			}

			return true;
		}

#pragma endregion  IO related

#pragma region Train Related
	public:
		/**
		 * @Synopsis BeginTrain Reset the optimizer to the initialization status of training
		 */
		virtual void BeginTrain() {
			OnlineModel<FeatType, LabelType>::BeginTrain();

			//reset weight vector
			for (int i = 0; i < this->classfier_num; ++i){
				this->weightMatrix[i].set_value(0);
			}
		}

		/**
		 * @Synopsis EndTrain called when a train ends
		 */
		virtual void EndTrain() {
			OnlineModel<FeatType, LabelType>::EndTrain();
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
				new_dim++; //reserve the 0-th
				for (int i = 0; i < this->classfier_num; ++i){
					s_array<float>& weightVec = this->weightMatrix[i];
					weightVec.reserve(new_dim);
					weightVec.resize(new_dim);
					//set the new value to zero
					weightVec.zeros(weightVec.begin + this->weightDim,
						weightVec.end);
				}
				this->weightDim = new_dim;
			}
		}

		/**
		 * @Synopsis SetParameter set parameters for the learning model
		 *
		 */
		virtual void SetParameter(BOC::Params &param){
			OnlineModel<FeatType, LabelType>::SetParameter(param);
		}

		/**
		 * @Synopsis IterateBC Iteration of online learning for binary classification
		 *
		 * @Param x current input data example
		 *
		 * @Returns  predicted class of the current example
		 */
		virtual int IterateBC(const DataPoint<FeatType, LabelType> &x, float* predict){
			this->curIterNum++;
			*predict = this->TrainPredict(*this->pWeightVecBC, x);
			int label = this->GetClassLabel(x);
			float gt = 0;
			this->lossFunc->GetGradient(label, predict, &gt);
			if (gt != 0){
				this->UpdateWeightVec(x, *this->pWeightVecBC, gt);
			}
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
		virtual int IterateMC(const DataPoint<FeatType, LabelType> &x, float* predict){
			return 0;
		}

	protected:
		/**
		 * @Synopsis TrainPredict prediction function for training
		 *
		 * @Param weightVec weight vector
		 * @Param data input data sample
		 *
		 * @Returns predicted value
		 */
		virtual float TrainPredict(const s_array<float> &weightVec, const DataPoint<FeatType, LabelType> &data) {
			float predict = 0;
			size_t dim = data.indexes.size();
			for (size_t i = 0; i < dim; i++){
				predict += weightVec[data.indexes[i]] * data.features[i];
			}
			predict += weightVec[0];
			return predict;
		}

		/**
		 * @Synopsis UpdateWeightVec Update the weight vector
		 *
		 * @Param x current input data example
		 * @Param weightVec weight vector to be updated
		 * @param gt common part of the gradient
		 *
		 */
		virtual void UpdateWeightVec(const DataPoint<FeatType, LabelType> &x, s_array<float>& weightVec, float gt) = 0;
#pragma endregion Train Related


#pragma region	Test related
	public:
		/**
		 * @Synopsis Predict prediction function for test
		 *
		 * @Param data input data sample
		 * @Param predicts predicted values for each classifier
		 *
		 * @Returns predicted class
		 */
		virtual int Predict(const DataPoint<FeatType, LabelType> &data, float* predicts){
			for (int k = 0; k < this->classfier_num; ++k){
				predicts[k] = this->TestPredict(this->weightMatrix[k], data);
			}
			if (this->classfier_num == 1){
				int label = this->GetClassLabel(data);
				if (this->IsCorrect(label, predicts) == false){
					return -label;
				}
				else{
					return data.label;
				}
			}
			else{
				return std::max_element(predicts, predicts + this->classfier_num) - predicts;
			}
		}

	protected:
		/**
		 * @Synopsis TestPredict prediction function for test
		 *
		 * @Param weightVec weight vector
		 * @Param data input data sample
		 *
		 * @Returns predicted value
		 */
		float TestPredict(const s_array<float> &weightVec, const DataPoint<FeatType, LabelType> &data) {
			float predict = 0;
			size_t dim = data.indexes.size();
			for (size_t i = 0; i < dim; i++){
				if (data.indexes[i] < this->weightDim){
					predict += weightVec[data.indexes[i]] * data.features[i];
				}
			}
			predict += weightVec[0];

			return predict;
		}

#pragma endregion	Test related

	};
}

#endif
