/*************************************************************************
> File Name: LossFunction.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/18 星期日 16:48:55
> Functions: base class for loss function
************************************************************************/

#pragma once
#include <cmath>
#include "../util.h"

namespace SOL
{
	template <typename FeatType, typename LabelType>
	class LossFunction
	{
        public:
		virtual bool IsCorrect(const DataPoint<FeatType, LabelType> &x, double predict) = 0;
        virtual double GetLoss(const DataPoint<FeatType, LabelType> &x, double predict) = 0;
        virtual double GetGradient(const DataPoint<FeatType, LabelType> &x, double predict, int index) = 0;
        virtual double GetBiasGradient(const DataPoint<FeatType, LabelType> &x, double predict) = 0;

        void GetGradient(const DataPoint<FeatType, LabelType> &x, double predict, double *gt) 
        {
            int dim = x.Dim();
            for (int i = 0; i < dim; i++)
                gt[i] = this->GetGradient(x,predict, i);
        }

    };
}
