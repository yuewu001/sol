/*************************************************************************
> File Name: FOBOS.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/20 Tuesday 11:14:54
> Functions: FOBOS: Efficient Online Batch Learning Using 
					Forward Backward Splitting

> Reference:
		Duchi J, Singer Y. Efficient online and batch learning using 
		forward backward splitting[J]. The Journal of Machine Learning 
		Research, 2009, 10: 2899-2934.

		This file implements the L1 and L2 square regularization
 ************************************************************************/

#pragma once
#include "../common/global.h"
#include "../common/util.h"

#include "Optimizer.h"


#include <cmath>
#include <limits>

namespace SOL {
	template <typename FeatType, typename LabelType>
	class FOBOS: public Optimizer<FeatType, LabelType> {
	public:
		FOBOS(DataSet<FeatType, LabelType> &dataset, 
			LossFunction<FeatType, LabelType> &lossFunc);
		~FOBOS();

	protected:
		//this is the core of different updating algorithms
		virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);

        //Change the dimension of weights
		virtual void UpdateWeightSize(IndexType newDim);

		//reset
		virtual void BeginTrain();

	protected:
		size_t *timeStamp;
	};

	template <typename FeatType, typename LabelType>
	FOBOS<FeatType, LabelType>::FOBOS(DataSet<FeatType, LabelType> &dataset, 
		LossFunction<FeatType, LabelType> &lossFunc):
	Optimizer<FeatType, LabelType>(dataset, lossFunc), timeStamp(NULL) {
		this->timeStamp = new size_t[this->weightDim];
	}

	template <typename FeatType, typename LabelType>
	FOBOS<FeatType, LabelType>::~FOBOS() {
		if(this->timeStamp != NULL)
			delete []this->timeStamp;
	}

    /**
	 *  UpdateWeightVec: use L1 norm as the regularization term
     *  this is the core of different updating algorithms
	 *  r(w) = lambda * |w|
	 *
	 * @tparam FeatType
	 * @tparam LabelType
	 * @Param:  x
	 * @Param:  y
	 */
	template <typename FeatType, typename LabelType>
	float FOBOS<FeatType,LabelType>::UpdateWeightVec (
            const DataPoint<FeatType, LabelType> &x) {
		float y = this->Predict(x);
		size_t featDim = x.indexes.size();
		float gt_i = this->eta * this->lossFunc->GetGradient(x.label,y);

        IndexType index_i = 0;
        float alpha = this->eta * this->lambda;
        for (size_t i = 0; i < featDim; i++) {
            index_i = x.indexes[i];
            //update the weight
            this->weightVec[index_i] -= gt_i * x.features[i];
			//lazy update
            int stepK = this->curIterNum - this->timeStamp[index_i];
            this->timeStamp[index_i] = this->curIterNum;

            this->weightVec[index_i] = trunc_weight(this->weightVec[index_i],
                    stepK * alpha);
        }
		//update bias term
		this->weightVec[0] -= gt_i;

		return y;
	}

	/**
	 *  UpdateWeightVec_L2S Use squre of L2 as the regularizatio term
	 *  r(w) = 0.5 * lambda * ||w||^2
	 *
	 * @tparam FeatType
	 * @tparam LabelType
	 * @Param:  x
	 * @Param:  y
	 */
    /*
	template <typename FeatType, typename LabelType>
	float FOBOS<FeatType,LabelType>::UpdateWeightVec_L2S(const DataPoint<FeatType, LabelType> &x)
	{
		int featDim = x.indexes.size();
        int index_i = 0;
		for (int i = 0; i < featDim; i++)
        {
            //lazy update
            index_i = x.indexes[i];
            int stepK = this->curIterNum - this->timeStamp[index_i];
            this->timeStamp[index_i] = this->curIterNum;
            this->weightVec[index_i] /= std::pow(1 + this->lambda,stepK);
        }

		float y = this->Predict(x);
        float gt_i = this->lossFunc->GetGradient(x.label,y);

		for (int i = 0; i < featDim; i++)
            this->weightVec[x.indexes[i]] -= this->eta * gt_i * x.features[i];		//update the weight

		//update bias term
		this->weightVec[0] -= this->eta * gt_i;
		return y;
	}
    */

	//reset the optimizer to this initialization
	template <typename FeatType, typename LabelType>
	void FOBOS<FeatType, LabelType>::BeginTrain() {
		Optimizer<FeatType, LabelType>::BeginTrain();
		//reset time stamp
		memset(this->timeStamp,0,sizeof(size_t) * this->weightDim);
	}

	//Change the dimension of weights
	template <typename FeatType, typename LabelType>
	void FOBOS<FeatType, LabelType>::UpdateWeightSize(IndexType newDim) {
		if (newDim < this->weightDim)
			return;
		else {
            newDim++;
			size_t* newT = new size_t[newDim];
            //copy info
			memcpy(newT,this->timeStamp,sizeof(size_t) * this->weightDim);
            //set the rest to zero
			memset(newT + this->weightDim,0,sizeof(size_t) * (newDim - this->weightDim));
			delete []this->timeStamp;
			this->timeStamp = newT;

			Optimizer<FeatType,LabelType>::UpdateWeightSize(newDim - 1);
		}
	}
}

