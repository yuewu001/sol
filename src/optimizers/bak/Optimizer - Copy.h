/*************************************************************************
> File Name: Optimizer.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/18 Sunday 16:04:18
> Functions: Base class for different algorithms to do machine learning
************************************************************************/

#ifndef HEADER_OPTIMIZER
#define HEADER_OPTIMIZER

#include "../io/DataSet.h"


/**
*  namespace: Machine Learning
*/
namespace ML{
	template <typename FeatType, typename LabelType> class Optimizer {
	protected:
        //input dataset
		DataSet<FeatType, LabelType> &dataSet;
        //number of iterations
		size_t update_times;

        inline size_t GetUpdateTimes() const { return this->update_times; }

	public:
		Optimizer(DataSet<FeatType, LabelType> &dataset);

		virtual ~Optimizer() {
		}

	protected:
		//train the data
		float Train();

		//try and get the best parameter
		virtual void BestParameter();

	public:
		//learn a model
		inline float Learn(int numOfTimes = 1);
		//learn a model and return the mistake rate and its variance
		float Learn(float &aveErrRate, float &varErrRate, float &sparseRate, int numOfTimes = 1);
		//test the performance on the given set
		float Test(DataSet<FeatType, LabelType> &testSet);

	};

    template <typename FeatType, typename LabelType>
	Optimizer<FeatType, LabelType>::Optimizer(DataSet<FeatType, LabelType> &dataset,
		LossFunction<FeatType, LabelType> &lossFunc) : dataSet(dataset) {
		this->weightDim = 1;
		//weight vector
		this->weightVec.resize(this->weightDim);

		this->eta0 = init_eta;;
		this->lambda = init_lambda;
		this->curIterNum = 0;
		this->initial_t = init_initial_t;
		this->power_t = init_power_t;

		this->sparse_soft_thresh = init_sparse_soft_thresh;
		this->is_normalize = false;
	}

	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void Optimizer<FeatType, LabelType>::BeginTrain() {
		//reset weight vector
		this->weightVec.set_value(0);
		this->curIterNum = this->initial_t;
		this->update_times = 0;
	}

	//called when a train ends
	template <typename FeatType, typename LabelType>
	void Optimizer<FeatType, LabelType>::EndTrain() {
		for (IndexType i = 1; i < this->weightDim; i++){
			if (this->weightVec[i] < this->sparse_soft_thresh &&
				this->weightVec[i] > -this->sparse_soft_thresh){
				this->weightVec[i] = 0;
			}
		}
	}

	template <typename FeatType, typename LabelType>
	float Optimizer<FeatType, LabelType>::Train() {
		if (dataSet.Rewind() == false)
			return 1.f;
		//reset
		this->BeginTrain();
		float errorNum(0);
		size_t show_step = 1; //show information every show_step
		size_t show_count = 2;
		size_t data_count = 0;

		//double train_time  = 0;
		printf("Iterate No.\t\tError Rate\t\t\n");
		while (1) {
			DataChunk<FeatType, LabelType> &chunk = dataSet.GetChunk();
			//double time1 = get_current_time();
			//all the data has been processed!
			if (chunk.dataNum == 0) {
				dataSet.FinishRead();
				break;
			}

            for (size_t i = 0; i < chunk.dataNum; i++) {
                DataPoint<FeatType, LabelType> &data = chunk.data[i];

                IndexType* p_index = data.indexes.begin;
                float* p_feat = data.features.begin;
                if (is_normalize){
                    if (data.sum_sq != 1){
                        float norm = sqrtf(data.sum_sq);
                        while (p_index != data.indexes.end){
                            *p_feat /= norm;
                            p_index++; p_feat++;
                        }
                    }
                }

                this->UpdateWeightSize(data.dim());
                float y = this->UpdateWeightVec(data);
                //loss
                if (this->lossFunc->IsCorrect(data.label, y) == false){
                    errorNum++;
                    data.margin = y * data.label;
                }
                data_count++;
                if (show_count == data_count){
                    printf("%lu\t\t\t%.6f\t\t\n", data_count,
                            errorNum / (float)(data_count));
                    show_count = (size_t(1) << ++show_step);
                }
                this->curIterNum++;
            }
			//double time2 = get_current_time();
			//train_time += time2 - time1;
			dataSet.FinishRead();
		}
		this->EndTrain();
		//cout<<"Purely Training Time: "<<train_time<<" s"<<endl;
		return errorNum / dataSet.size();
	}

	//learn a model and return the mistake rate and its variance
	template <typename FeatType, typename LabelType>
	float Optimizer<FeatType, LabelType>::Learn(float &aveErrRate, float &varErrRate,
		float &sparseRate, int numOfTimes) {
		float * errorRateVec = new float[numOfTimes];
		float * sparseRateVec = new float[numOfTimes];

		for (int i = 0; i < numOfTimes; i++) {
			//random order
			errorRateVec[i] = this->Train();
			sparseRateVec[i] = this->GetSparseRate();
		}
		aveErrRate = Average(errorRateVec, numOfTimes);
		varErrRate = Variance(errorRateVec, numOfTimes);
		sparseRate = Average(sparseRateVec, numOfTimes);

		delete[]errorRateVec;
		delete[]sparseRateVec;

		return aveErrRate;
	}

	//learn a model
	template <typename FeatType, typename LabelType>
	float Optimizer<FeatType, LabelType>::Learn(int numOfTimes) {
		float aveErrRate, varErrRate, sparseRate;
		return this->Learn(aveErrRate, varErrRate, sparseRate, numOfTimes);
	}

	//test the performance on the given set
	template <typename FeatType, typename LabelType>
	float Optimizer<FeatType, LabelType>::Test(DataSet<FeatType, LabelType> &testSet) {
		if (testSet.Rewind() == false)
			return 1.f;
		float errorRate(0);
		//double test_time = 0;
		//test
		while (1) {
			const DataChunk<FeatType, LabelType> &chunk = testSet.GetChunk(true);
			//double time1 = get_current_time();
			if (chunk.dataNum == 0) //"all the data has been processed!"
				break;
			for (size_t i = 0; i < chunk.dataNum; i++) {
				const DataPoint<FeatType, LabelType> &data = chunk.data[i];
				IndexType* p_index = data.indexes.begin;
				float* p_feat = data.features.begin;
				if (is_normalize){
					if (data.sum_sq != 1){
						float norm = sqrtf(data.sum_sq);
						while (p_index != data.indexes.end){
							*p_feat /= norm;
							p_index++; p_feat++;
						}
					}
				}
				//predict
				float predict = this->Test_Predict(data);
				if (this->lossFunc->IsCorrect(data.label, predict) == false)
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

	template <typename FeatType, typename LabelType>
	float Optimizer<FeatType, LabelType>::Test_Predict(const DataPoint<FeatType, LabelType> &data) {
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
	template <typename FeatType, typename LabelType>
	float Optimizer<FeatType, LabelType>::Predict(const DataPoint<FeatType, LabelType> &data) {
		float predict = 0;
		IndexType* p_index = data.indexes.begin;
		float* p_feat = data.features.begin;
		while (p_index != data.indexes.end){
			predict += this->weightVec[*p_index++] * (*p_feat++);
		}
		predict += this->weightVec[0];
		return predict;
	}

	template <typename FeatType, typename LabelType>
	IndexType Optimizer<FeatType, LabelType>::GetNonZeroNum(){
		IndexType non_zeroNum(0);
		if (this->weightDim == 1)
			return 0;

		for (IndexType i = 1; i < this->weightDim; i++) {
			if (this->weightVec[i] != 0)
				non_zeroNum++;
		}
		return non_zeroNum;
	}

	template <typename FeatType, typename LabelType>
	float Optimizer<FeatType, LabelType>::GetSparseRate(IndexType total_len) {
		if (this->weightDim == 1)
			return 1;
		IndexType zeroNum = this->weightDim - 1 - this->GetNonZeroNum();

		if (total_len > 0)
			return zeroNum / (float)total_len;
		else
			return zeroNum / (float)(this->weightDim - 1);
	}

	//try and get the best parameter
	template <typename FeatType, typename LabelType>
	void Optimizer<FeatType, LabelType>::BestParameter() {
		float prev_lambda = this->lambda;
		this->lambda = 0;
		//1. Select the best eta0

		float min_errorRate = 1;
		float bestEta = 1;

		for (float eta_c = init_eta_min; eta_c <= init_eta_max; eta_c *= init_eta_step) {
			cout << "eta0 = " << eta_c << "\n";
			float errorRate(0);
			this->eta0 = eta_c;
			errorRate += this->Train();

			if (errorRate < min_errorRate) {
				bestEta = eta_c;
				min_errorRate = errorRate;
			}
			cout << "mistake rate: " << errorRate * 100 << " %\n";
		}
		this->eta0 = bestEta;
		this->lambda = prev_lambda;
		cout << "Best Parameter:\teta = " << this->eta0 << "\n\n";
	}

	template <typename FeatType, typename LabelType>
	void Optimizer<FeatType, LabelType>::SetParameter(float lambda, float eta0,
		float power_t, int t0){
		this->lambda = lambda >= 0 ? lambda : this->lambda;
		this->eta0 = eta0 > 0 ? eta0 : this->eta0;
		this->power_t = power_t >= 0 ? power_t : this->power_t;
		this->initial_t = t0 > 0 ? t0 : this->initial_t;
	}

	//Change the dimension of weights
	template <typename FeatType, typename LabelType>
	void Optimizer<FeatType, LabelType>::UpdateWeightSize(IndexType newDim) {
		if (newDim < this->weightDim)
			return;
		else {
			newDim++; //reserve the 0-th
			this->weightVec.reserve(newDim);
			this->weightVec.resize(newDim);
			//set the new value to zero
			this->weightVec.zeros(this->weightVec.begin + this->weightDim,
				this->weightVec.end);
			this->weightDim = newDim;
		}
	}
	template <typename FeatType, typename LabelType>
	void Optimizer<FeatType, LabelType>::SaveModel(const string& filename){
		ofstream outfile(filename.c_str(), ios::out | ios::binary);
		if (!outfile){
			cerr << "open file " << filename << "failed!" << endl;
			return;
		}
		outfile << "y = b + w x\n";
		for (IndexType i = 0; i < this->weightDim; i++){
			outfile << this->weightVec[i] << "\n";
		}
		outfile.close();
	}
}
#endif
