/*************************************************************************
	> File Name: HingeLoss.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2013/8/18 星期日 16:58:22
	> Functions: Hinge Loss function, for SVM
 ************************************************************************/

#pragma once
#include "LossFunction.h"

namespace SOL
{
	template <typename FeatType, typename LabelType>
	class HingeLoss: public LossFunction<FeatType, LabelType>
	{
		public:
			virtual  double GetLoss(const DataPoint<FeatType, LabelType> &x, double predict)
			{
                return std::max(0.0, 1 - predict * x.label);
			}

			virtual  double GetGradient(const DataPoint<FeatType, LabelType> &x, double predict)
			{
                if (this->GetLoss(x,predict) > 0)
                    return -x.label;
                else
					return 0;
			}
	};
}
