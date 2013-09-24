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
			virtual  bool IsCorrect(const DataPoint<FeatType, LabelType> &x, double predict)
			{
				return Sgn(predict) == x.Label() ? true : false;
			} 

			virtual  double GetLoss(const DataPoint<FeatType, LabelType> &x, double predict)
			{
				if (this->IsCorrect(x,predict) == false)
					return std::max(0.0, 1 - predict * x.Label());
				else
					return 0;
			}

			virtual  double GetGradient(const DataPoint<FeatType, LabelType> &x, double predict, int index)
			{
				if (this->IsCorrect(x,predict) == true)
					return 0;
				return - x.Label() * x[index];
			}
			virtual  double GetBiasGradient(const DataPoint<FeatType, LabelType> &x, double predict)
			{
				if (this->IsCorrect(x,predict) == true)
					return 0;
				return -x.Label();
			}
	};
}
