/*************************************************************************
> File Name: Optimizer.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/18 星期日 16:04:18
> Functions: Base class for different algorithms to do online learning
************************************************************************/

#pragma once
#include "../data/DataSet.h"
#include "../loss/LossFunction.h"

#include <algorithm>
#include <numeric>

/**
*  namespace: Sparse Online Learning
*/
namespace SOL
{
	template <typename FeatType, typename LabelType>
	class Optimizer
	{
	protected:
		DataSet<FeatType, LabelType> &dataSet;

		//weight vector
	protected:
		double * weightVec;
		//weight dimenstion: can be the same to feature, or with an extra
		//bias
		int weightDim;

	protected:
		//pass number
		int passNum;
		double passDecay;
		//iteration number
		size_t curIterNum;

		//parameters
		double lambda;

		double eta; //learning rate
	protected:
		double mistakeRate;

	protected:
		LossFunction<FeatType, LabelType> *lossFunc;

	public:
		Optimizer(DataSet<FeatType, LabelType> &dataset, LossFunction<FeatType, LabelType> &lossFunc);

		virtual ~Optimizer()
		{
			if (weightVec != NULL)
				delete []this->weightVec;
		}

	protected:
		//reset the optimizer to this initialization
		virtual void Reset();
		//called when a pass ended
		virtual void PassEnd() = 0; 
		//called when a round ended
		virtual void RoundEnd() = 0; 

		//run the data one time
		double RunOnce();

		//this is the core of different updating algorithms
		//return the predict
		virtual double UpdateWeightVec(const DataPoint<FeatType, LabelType> &x) = 0;
		//predict a new feature
		virtual double Predict(const DataPoint<FeatType, LabelType> &data);

	public:
		void SetParameter(double lambda = -1, double eta = -1);
		//try and get the best parameter
		virtual void BestParameter(); 

	public:
		//learn a model
		inline double Learn(int numOfTimes = 20);
		//learn a model and return the mistake rate and its variance
		double Learn(double &aveErrRate, double &varErrRate, double &sparseRate, int numOfTimes = 20);
		//test the performance on the given set
		double Test(DataSet<FeatType, LabelType> &testSet);

		//learn and test the performance
		void LearnAndTest(DataSet<FeatType, LabelType> &testSet, 
			double &errRate, double &varRate, double&sparseRate, int numOfTime = 20);
		void LearnAndTest(DataSet<FeatType, LabelType> &testSet, int numOfTime = 20);

		double GetSparseRate();

	public:
		//Change the dimension of weights
		virtual void UpdateWeightSize(int newDim);
	};

	template <typename FeatType, typename LabelType>
	Optimizer<FeatType, LabelType>::Optimizer(DataSet<FeatType, LabelType> &dataset, 
		LossFunction<FeatType, LabelType> &lossFunc): dataSet(dataset)
	{
		this->eta = 1e-3;
		this->lambda = 0.1;

		this->passDecay = 1;
		this->passNum = 1;
		this->curIterNum = 0;


		this->lossFunc = &lossFunc;
		this->weightDim = dataset.Dim() + 1;
		this->weightVec = new double[this->weightDim];

		//weight vector
		memset(this->weightVec,0,sizeof(double) * this->weightDim);
		this->curIterNum = 0;

		this->dataSet.Bind(this);
	}

	template <typename FeatType, typename LabelType>
	double Optimizer<FeatType, LabelType>::RunOnce()
	{
		double errorNum(0);
		int dataId = 0;
		//reset
		this->Reset();

		for (int pass = 0; pass < this->passNum; pass++)
		{
			this->dataSet.Rewind();
			dataId = 0;
			//train
			while(true)
			{
				this->curIterNum++;
				DataPoint<FeatType, LabelType>& curData = this->dataSet.GetData(dataId);
				if (curData.data() == NULL)
					break;
				double y = this->UpdateWeightVec(curData); 
				//loss
				if (this->lossFunc->IsCorrect(curData,y) == false)
					errorNum++;
				dataId++;
			}
			this->PassEnd();
		}
		this->RoundEnd();

		errorNum = errorNum / (dataId * this->passNum);
		return errorNum;
	}

	//learn a model and return the mistake rate and its variance
	template <typename FeatType, typename LabelType>
	double Optimizer<FeatType, LabelType>::Learn(double &aveErrRate, double &varErrRate, double &sparseRate,
		int numOfTimes)
	{
		cout<<"\nLearning the model...\n";

		double* errorRateVec = new double[numOfTimes];
		double * sparseRateVec = new double[numOfTimes];

		for (int i = 0; i < numOfTimes; i++)
		{
			//random order
			this->dataSet.RandomOrder();
			errorRateVec[i] = this->RunOnce();
			sparseRateVec[i] = this->GetSparseRate();
		}
		aveErrRate = Average(errorRateVec, numOfTimes);
		varErrRate = Variance(errorRateVec, numOfTimes);
		sparseRate = Average(sparseRateVec, numOfTimes);

		delete []errorRateVec;
		delete []sparseRateVec;

		return aveErrRate;
	}

	//learn a model
	template <typename FeatType, typename LabelType>
	double Optimizer<FeatType, LabelType>::Learn(int numOfTimes)
	{
		double aveErrRate, varErrRate, sparseRate;
		return this->Learn(aveErrRate, varErrRate,sparseRate, numOfTimes);
	}

	//learn and test the performance
	template <typename FeatType, typename LabelType>
	void Optimizer<FeatType, LabelType>::LearnAndTest(DataSet<FeatType, LabelType> &testSet, 
		int numOfTimes)
	{
		double aveErrRate(0), varErrRate(0), sparseRate(0);
		this->LearnAndTest(testSet,aveErrRate, varErrRate,sparseRate,numOfTimes);
	}

	//learn and test the performance
	template <typename FeatType, typename LabelType>
	void Optimizer<FeatType, LabelType>::LearnAndTest(DataSet<FeatType, LabelType> &testSet, 
		double &aveErrRate, double &varErrRate, double& sparseRate, int numOfTimes)
	{
		cout<<"Learn and Test the model...\n";

		double * errorRateVec = new double[numOfTimes];
		double * sparseRateVec = new double[numOfTimes];
		memset(errorRateVec,0,sizeof(double) * numOfTimes);
		memset(sparseRateVec,0,sizeof(double) * numOfTimes);
		for (int i = 0; i < numOfTimes; i++)
		{
			//random order
			this->dataSet.RandomOrder();
			
			//train
			errorRateVec[i] = this->RunOnce();
			sparseRateVec[i] = this->GetSparseRate();
			//test
			errorRateVec[i] = this->Test(testSet);
		}

		aveErrRate = Average(errorRateVec, numOfTimes);
		varErrRate = Variance(errorRateVec, numOfTimes);
		sparseRate = Average(sparseRateVec, numOfTimes);
		

		delete []errorRateVec;
		delete []sparseRateVec;
	}

	//test the performance on the given set
	template <typename FeatType, typename LabelType>
	double Optimizer<FeatType, LabelType>::Test(
		DataSet<FeatType, LabelType> &testSet)
	{
		//test
		double errorRate(0);
		int dataId(0);
		DataPoint<FeatType, LabelType> curData; 
		//test
		testSet.Rewind();
		while(true)
		{
			DataPoint<FeatType, LabelType>& curData = testSet.GetData(dataId);
			if(curData.data() == NULL) //invalid data
				break;
			//predict
			double predict = this->Predict(curData);
			if (this->lossFunc->IsCorrect(curData,predict) == false)
				errorRate++;
			dataId++;
		}
		errorRate /= dataId;
		return errorRate;
	}


	template <typename FeatType, typename LabelType>
	double Optimizer<FeatType, LabelType>::Predict(const DataPoint<FeatType, LabelType> &data)
	{
		double predict = 0;
		int featDim = data.Dim();
		for (int i = 0; i < featDim; i++)
			predict += this->weightVec[i] * data[i];
		predict += this->weightVec[this->weightDim - 1];
		return predict;
	}

	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void Optimizer<FeatType, LabelType>::Reset()
	{
		//reset weight vector
		memset(this->weightVec,0,sizeof(double) * this->weightDim);
		this->curIterNum = 0;
	}

	template <typename FeatType, typename LabelType>
	double Optimizer<FeatType, LabelType>::GetSparseRate()
	{
		double zeroNum(0);
		for (int i = 0; i < this->weightDim - 1; i++)
		{
			if (this->weightVec[i] == 0)
				zeroNum++;
		}
		return zeroNum / (this->weightDim - 1);
	}

	//try and get the best parameter
	template <typename FeatType, typename LabelType>
	void Optimizer<FeatType, LabelType>::BestParameter()
	{
		//0. set sparsification to zero 
		double tmp_lambda = this->lambda;
		this->lambda = 0;

		//1. Select the best eta
		double eta_min = 1e-10;
		double eta_max = 1;

		float min_errorRate = 1;
		double bestEta = 1;

		this->dataSet.RandomOrder();
		for (double eta_c = eta_min; eta_c<= eta_max; eta_c *= 10)
		{
			cout<<"eta = "<<eta_c<<"\t";
			float errorRate(0);
			//for (int k = 1; k < 10; k++)
			{
				this->eta = eta_c;
				errorRate += this->RunOnce();
			}

			//errorRate /= 10;
			if (errorRate < min_errorRate)
			{
				bestEta = eta_c;
				min_errorRate = errorRate;
			}
			cout<<"mistake rate: "<<errorRate * 100<<" %\n";
		}
		this->eta = bestEta;
		this->lambda = tmp_lambda;
		cout<<"Best Parameter:\n\teta = "<<this->eta<<"\n\n";
	}

	template <typename FeatType, typename LabelType>
	void Optimizer<FeatType, LabelType>::SetParameter(double lambda , double eta)
	{
		this->lambda  = lambda >= 0 ? lambda : this->lambda;
		this->eta = eta > 0 ? eta : this->eta;
	}

	//Change the dimension of weights
	template <typename FeatType, typename LabelType>
	void Optimizer<FeatType, LabelType>::UpdateWeightSize(int newDim)
	{
		if (newDim <= this->weightDim - 1)
			return;
		else
		{
			double* newW = new double[newDim + 1];
			memset(newW,0,sizeof(double) * (newDim + 1));

			//copy info
			memcpy(newW,this->weightVec,sizeof(double) * (this->weightDim - 1));
			newW[newDim] = this->weightVec[this->weightDim - 1];

			delete []this->weightVec;
			this->weightVec = newW;
			this->weightDim = newDim + 1;
		}
	}
}