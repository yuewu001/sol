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

/**
*  namespace: Batch and Online Classification
*/
namespace BOC {
    template <typename FeatType, typename LabelType> 
    class OnlineLinearModel : public OnlineModel<FeatType, LabelType> {
        //weight vector
        protected:
            //the first element is zero
            s_array<float> weightVec;
            //weight dimension: can be the same to feature, or with an extra bias
            IndexType weightDim;

        public:
            OnlineLinearModel(LossFunction<FeatType, LabelType> *lossFunc) : OnlineModel<FeatType, LabelType>(lossFunc) {
                this->weightDim = 1;
                //weight vector
                this->weightVec.resize(this->weightDim);
            }

            virtual ~OnlineLinearModel() {
            }

            /**
             * @Synopsis inherited functions
             */
        public:
            /**
             * PrintOptInfo print the info of optimization algorithm
             */
            virtual void PrintOptInfo() const {
                OnlineModel<FeatType, LabelType>::PrintOptInfo();

                printf("Linear Model: y = w * x + b \n");
            }

            /**
             * @Synopsis BeginTrain Reset the optimizer to the initialization status of training
             */
            virtual void BeginTrain() {
                OnlineModel<FeatType, LabelType>::BeginTrain();

                //reset weight vector
                this->weightVec.set_value(0);
            }

            /**
             * @Synopsis EndTrain called when a train ends
             */
            virtual void EndTrain() { }

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
                    this->weightVec.reserve(new_dim);
                    this->weightVec.resize(new_dim);
                    //set the new value to zero
                    this->weightVec.zeros(this->weightVec.begin + this->weightDim,
                            this->weightVec.end);
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
			 * @Synopsis Test_Predict prediction function for test
			 *
			 * @Param data input data sample
			 *
			 * @Returns predicted value
			 */
			virtual float Test_Predict(const DataPoint<FeatType, LabelType> &data) {
				float predict = 0;
				size_t dim = data.indexes.size();
				IndexType index_i;
				for (size_t i = 0; i < dim; i++){
					index_i = data.indexes[i];
					if (index_i < this->weightDim && this->weightVec[index_i] != 0)
						predict += this->weightVec[index_i] * data.features[i];
				}
				predict += this->weightVec[0];
				return predict;
			}

			/**
			 * @Synopsis Predict prediction function for training
			 *
			 * @Param data input data sample
			 *
			 * @Returns predicted value
			 */
			virtual float Predict(const DataPoint<FeatType, LabelType> &data) {
				float predict = 0;
				size_t dim = data.indexes.size();
				for (size_t i = 0; i < dim; i++){
					predict += this->weightVec[data.indexes[i]] * data.features[i];
				}
				predict += this->weightVec[0];
				return predict;
			}

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
			os << "weight dimension: " << this->weightDim << "\n";
			//weights
			for (IndexType i = 0; i < this->weightDim; i++){
				if (this->weightVec[i] != 0){
					os << i << ":" << this->weightVec[i] << "\n";
				}
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
			is >> this->weightDim;
			getline(is, line);

			//weights
			for (IndexType i = 0; i < this->weightDim; i++){
				is >> this->weightVec[i];
			}

			return true;
		}

	};
}

#endif
