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

static const int init_weight_dim = 1;
/**
*  namespace: Sparse Online Learning
*/
namespace SOL
{
	template <typename FeatType, typename LabelType> class Optimizer
	{
        //Iteration 
        protected:
            //iteration number
            size_t curIterNum;
            //parameters
            double lambda;
            double eta; //learning rate
            double mistakeRate;

            DataSet<FeatType, LabelType> &dataSet;

            //weight vector
        protected:
            //the first element is zero
            double *weightVec;
            //weight dimenstion: can be the same to feature, or with an extra bias
            int weightDim;

            //For sparse
        protected:
            double sparse_soft_thresh;

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
            //reset the optimizer to this initialization at the beginning of
            //training
            virtual void BeginTrain();
            //called when a train ends
            virtual void EndTrain(){}
            //train the data
            double Train();
            //predict a new feature
            double Predict(const DataPoint<FeatType, LabelType> &data);

            //this is the core of different updating algorithms
            //return the predict
            virtual double UpdateWeightVec(const DataPoint<FeatType, LabelType> &x) = 0;

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

        protected:
            //Change the dimension of weights
            virtual void UpdateWeightSize(int newDim);
    };

    template <typename FeatType, typename LabelType>
        Optimizer<FeatType, LabelType>::Optimizer(DataSet<FeatType, LabelType> &dataset, 
                LossFunction<FeatType, LabelType> &lossFunc): dataSet(dataset)
    {
        
        this->lossFunc = &lossFunc;
        this->weightDim = init_weight_dim;
        //weight vector
        this->weightVec = new double[this->weightDim];

        this->eta = 1e-3;
        this->lambda = 10;
        this->curIterNum = 0;

        this->sparse_soft_thresh = 0;
    }

    //reset the optimizer to this initialization
    template <typename FeatType, typename LabelType>
        void Optimizer<FeatType, LabelType>::BeginTrain()
        {
            //reset weight vector
            memset(this->weightVec,0,sizeof(double) * this->weightDim);
            this->curIterNum = 0;
        }
   
    template <typename FeatType, typename LabelType> 
        double Optimizer<FeatType, LabelType>::Train()
        {
            if(dataSet.Rewind() == false)
                exit(0);
            //reset
            this->BeginTrain();
            double errorNum(0);
            while(1)
            {
                const DataChunk<FeatType,LabelType> &chunk = dataSet.GetChunk();
                //all the data has been processed!
                if(chunk.dataNum  == 0) 
                    break;

                for (size_t i = 0; i < chunk.dataNum; i++)
                {
                    this->curIterNum++;
                    const DataPoint<FeatType, LabelType> &data = chunk.data[i];
                    this->UpdateWeightSize(data.max_index);
                    double y = this->UpdateWeightVec(data); 
                    //loss
                    if (this->lossFunc->IsCorrect(data,y) == false)
                        errorNum++;
                }
                dataSet.FinishRead();
            }
            this->EndTrain();

            return errorNum / dataSet.size();
        }

    //learn a model and return the mistake rate and its variance
    template <typename FeatType, typename LabelType>
        double Optimizer<FeatType, LabelType>::Learn(double &aveErrRate, double &varErrRate, 
                double &sparseRate, int numOfTimes)
        {
            cout<<"\nLearning the model...\n";

            double* errorRateVec = new double[numOfTimes];
            double * sparseRateVec = new double[numOfTimes];

            for (int i = 0; i < numOfTimes; i++)
            {
                //random order
                errorRateVec[i] = this->Train();
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
                //train
                errorRateVec[i] = this->Train();
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
        double Optimizer<FeatType, LabelType>::Test(DataSet<FeatType, LabelType> &testSet)
        {
            double errorRate(0);
            //test
            while(1)
            {
                const DataChunk<float,char> &chunk = testSet.GetChunk();
                if(chunk.dataNum  == 0) //"all the data has been processed!"
                    break;
                for (int i = 0; i < chunk.dataNum; i++)
                {
                    const DataPoint<float,char> &data = chunk.data[i];
                    //predict
                    double predict = this->Predict(data);
                    if (this->lossFunc->IsCorrect(data,predict) == false)
                        errorRate++;
                }
                testSet.FinishRead();
            }
            errorRate /= testSet.size();
            return errorRate;
        }


    template <typename FeatType, typename LabelType>
        double Optimizer<FeatType, LabelType>::Predict(const DataPoint<FeatType, LabelType> &data)
        {
            double predict = 0;
            int dim = data.indexes.size();
            for (int i = 0; i < dim; i++)
                predict += this->weightVec[data.indexes[i]] * data.features[i];
            predict += this->weightVec[0];
            return predict;
        }


    template <typename FeatType, typename LabelType>
        double Optimizer<FeatType, LabelType>::GetSparseRate()
        {
            double zeroNum(0);
            for (int i = 0; i < this->weightDim; i++)
            {
                if (this->weightVec[i] <= this->sparse_soft_thresh && 
                        this->weightVec[i] >= -this->sparse_soft_thresh)
                    zeroNum++;
            }
            return zeroNum / this->weightDim;
        }

    //try and get the best parameter
    template <typename FeatType, typename LabelType>
        void Optimizer<FeatType, LabelType>::BestParameter()
        {
            //1. Select the best eta
            double eta_min = 1e-10;
            double eta_max = 1;

            float min_errorRate = 1;
            double bestEta = 1;

            for (double eta_c = eta_min; eta_c<= eta_max; eta_c *= 10)
            {
                cout<<"eta = "<<eta_c<<"\t";
                float errorRate(0);
                this->eta = eta_c;
                errorRate += this->Train();

                if (errorRate < min_errorRate)
                {
                    bestEta = eta_c;
                    min_errorRate = errorRate;
                }
                cout<<"mistake rate: "<<errorRate * 100<<" %\n";
            }
            this->eta = bestEta;
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
            if (newDim < this->weightDim) 
                return;
            else
            {
                newDim++; //reserve the 0-th
                double* newW = new double[newDim];
                memset(newW,0,sizeof(double) * newDim); 

                //copy info
                memcpy(newW,this->weightVec,sizeof(double) * this->weightDim); 
                //set the rest to zero
                memset(newW + this->weightDim,0,sizeof(double) * (newDim - this->weightDim));

                delete []this->weightVec;
                this->weightVec = newW;
                this->weightDim = newDim;
            }
        }
}
