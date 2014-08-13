/*************************************************************************
> File Name: Optimizer.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/18 Sunday 16:04:18
> Functions: Base class for different algorithms to do machine learning
************************************************************************/

#ifndef HEADER_OPTIMIZER
#define HEADER_OPTIMIZER

#include "../io/DataSet.h"
#include "../algorithms/LearnModel.h"

#include "../utils/reflector.h"


#include <fstream>

/**
*  namespace: Batch and Online Classification
*/
namespace BOC{

	template <typename FeatType, typename LabelType>
	class Optimizer : public Registry {

	protected:
		typedef DataPoint<FeatType, LabelType> PointType;

	protected:
		//learning model
		LearnModel<FeatType, LabelType> *learnModel;
		//input dataset
		DataSet<FeatType, LabelType> *dataSet;
		//number of iterations
		size_t update_times;

		//pre-selected features
		s_array<char> sel_feat_flag_vec;
		IndexType max_index;


		/**
		 * @Synopsis Constructors
		 */
	public:
		Optimizer(LearnModel<FeatType, LabelType> *model, DataSet<FeatType, LabelType> *dataset) :
			learnModel(model), dataSet(dataset) {
			this->update_times = 0;
			this->max_index = 0;
		}

		virtual ~Optimizer() {
		}

		/**
		 * @Synopsis Interfaces
		 */
	public:
		//train the data
		virtual float Train() = 0;

		/**
		 * @Synopsis Test test the performance on the given set
		 *
		 * @Param testSet
		 *
		 * @Returns
		 */
		float Test(DataSet<FeatType, LabelType> &testSet) {
			testSet.Rewind();
			float errorRate(0);
			//double test_time = 0;
			//test
			float* predicts = new float[this->learnModel->GetClassfierNum()];
			while (1) {
				const DataChunk<PointType> &chunk = testSet.GetChunk();
				//double time1 = get_current_time();
				if (chunk.dataNum == 0) //"all the data has been processed!"
					break;
				for (size_t i = 0; i < chunk.dataNum; i++) {
					const PointType &data = chunk.data[i];
					//predict
					int predict = this->learnModel->Predict(data, predicts);
					if (predict != data.label)
						errorRate++;
				}
				//double time2 = get_current_time();
				//test_time += time2 - time1;
				testSet.FinishRead();
			}
			//printf("accumulated test time %lf ms\n",test_time);
			errorRate /= testSet.size();
			return errorRate;
		}

		/**
		 * @Synopsis Test test the performance on the given set
		 *
		 * @Param testSet
		 * @Param os output stream to save the predicted values
		 *
		 * @Returns
		 */
		float Test(DataSet<FeatType, LabelType> &testSet, std::ostream& os) {
			testSet.Rewind();
			float errorRate(0);
			//double test_time = 0;
			//test
			float* predicts = new float[this->learnModel->GetClassfierNum()];
			while (1) {
				const DataChunk<PointType> &chunk = testSet.GetChunk();
				//double time1 = get_current_time();
				if (chunk.dataNum == 0) //"all the data has been processed!"
					break;
				for (size_t i = 0; i < chunk.dataNum; i++) {
					const PointType &data = chunk.data[i];
					//predict
					int predict = this->learnModel->Predict(data, predicts);
					os << predict << "\t" << (int)(data.label) << "\n";
					if (predict != data.label){
						errorRate++;
					}
				}
				//double time2 = get_current_time();
				//test_time += time2 - time1;
				testSet.FinishRead();
			}
			//printf("accumulated test time %lf ms\n",test_time);
			errorRate /= testSet.size();
			return errorRate;
		}

		/**
		 * @Synopsis GetUpdateTimes get the number of iterations
		 *
		 * @Returns number of iterations
		 */
		inline size_t GetUpdateTimes() const { return this->update_times; }

	protected:
		/**
		 * @Synopsis Reset reset the optimizer
		 *
		 * @Returns  true if reset successfully
		 */
		bool Reset() {
			this->update_times = 0;
			this->dataSet->Rewind();
			return true;
		}

	public:
		/**
		 * @Synopsis load the feature selection result
		 *
		 * @Returns  error code, zero if sucussful
		 */
		int LoadPreSelFeatures(const string& filename){
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
			char* line = (char*)malloc(sizeof(char) * dst_len);
			while (io_handler.read_line(line, dst_len) != NULL){
				line = strip_line(line);
				//skip comments and empty lines
				if (line[0] == '\0' || line[0] == '#')
					continue;

				featIndex = (IndexType)(atoi(line));
				if (featIndex == 0){
					fprintf(stderr, "parse index %s failed!\n", line);
					err_code = 1;
					break;
				}
				featVec.push_back(featIndex);
			}
			if (line != NULL)
				free(line);

			//find the max index
			this->max_index = *std::max_element(featVec.begin(), featVec.end());
			this->sel_feat_flag_vec.resize(this->max_index + 1);
			this->sel_feat_flag_vec.zeros();

			for (vector<IndexType>::iterator iter = featVec.begin(); iter != featVec.end(); ++iter){
				this->sel_feat_flag_vec[*iter] = 1;
			}

			return err_code;
		}

	protected:
		/**
		 * @Synopsis load the feature selection result
		 *
		 * @Returns  error code, zero if sucussful
		 */
		void FilterFeatures(PointType& data){
			if (this->max_index == 0){
				return;
			}
			size_t featNum = data.indexes.size();
			for (size_t i = 0; i < featNum; ++i){
				if (!(data.indexes[i] <= this->max_index && this->sel_feat_flag_vec[data.indexes[i]] != 0)){
					data.features[i] = 0;
				}
			}
		}
	};
}

#endif
