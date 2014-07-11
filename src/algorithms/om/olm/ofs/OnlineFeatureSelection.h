/*************************************************************************
	> File Name: OnlineFeatureSelection.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 7/11/2014 11:45:56 AM
	> Functions: base class for online feature selection
 ************************************************************************/

#ifndef HEADER_ONLINE_FEATURE_SELECTION
#define HEADER_ONLINE_FEATURE_SELECTION

/**
*  namespace: Batch and Online Classification
*/
namespace BOC {
	template <typename FeatType, typename LabelType>
	class OnlineFeatureSelection : public OnlineLinearModel<FeatType, LabelType> {
	protected:
        //keep top K elemetns
		IndexType K; 

	public:
		OnlineFeatureSelection(LossFunction<FeatType, LabelType> *lossFunc) :
			OnlineLinearModel<FeatType, LabelType>(lossFunc) {
                this->K = 1;
			}

		virtual ~OnlineFeatureSelection() {
		}

		//inherited functions
	public:
		/**
		 * PrintModelSettings print the info of optimization algorithm
		 */
		virtual void PrintModelSettings() const {
			OnlineLinearModel<FeatType, LabelType>::PrintModelSettings();

			printf("Linear Online Feature Selection:\n");
			printf("\tK:\t%d\n", this->K);
		}

		/**
		 * PrintModelInfo print the info of trained model
		 */
		virtual void PrintModelInfo() const {
			OnlineLinearModel<FeatType, LabelType>::PrintModelInfo();

			IndexType zeroNum = 0;
			if (this->weightDim == 1){
				zeroNum = 1;
			}
			else{
				zeroNum = this->weightDim - 1 - this->GetNonZeroNum();
			}

			double sparseRate = zeroNum / (double)(this->weightDim - 1);
			printf("Sparsification Rate: %g %%\n", sparseRate * 100);
		}

		/**
		 * @Synopsis SetParameter set parameters for the learning model
		 *
		 * @Param param
		 */
		virtual void SetParameter(BOC::Params &param){
			OnlineLinearModel<FeatType, LabelType>::SetParameter(param);
			this->K = param.IntValue("-k");
            INVALID_ARGUMENT_EXCEPTION(K, this->K > 0, "larger than 0");
		}

		/**
		 * @Synopsis BeginTrain Reset the optimizer to the initialization status of training
		 */
		virtual void BeginTrain() {
			OnlineLinearModel<FeatType, LabelType>::BeginTrain();
		}

		/**
		 * @Synopsis EndTrain called when a train ends
		 */
		virtual void EndTrain() { 
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

			//select k features
            os << "K : " << this->K<< "\n";

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

			//select k features
			string line;
			getline(is, line, ':');
			is >> this->K;
			getline(is, line);

			return true;
		}
	};
}
#endif

