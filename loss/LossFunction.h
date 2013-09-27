/*************************************************************************
> File Name: LossFunction.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/18 星期日 16:48:55
> Functions: base class for loss function
************************************************************************/

#pragma once
#include <cmath>
#include "../common/util.h"

namespace SOL
{
	template <typename FeatType, typename LabelType>
	class LossFunction
	{
        char Sign(double x)
        {
            if (x >= 0) 
                return 1;
            else
                return -1;
        }

        public:
		bool IsCorrect(const DataPoint<FeatType, LabelType> &x, double predict)
        {
            return this->Sign(predict) == x.label ? true : false;
        }

        virtual double GetLoss(const DataPoint<FeatType, LabelType> &x, double predict) = 0;
        virtual double GetGradient(const DataPoint<FeatType, LabelType> &x, double predict) = 0;
    };
}
