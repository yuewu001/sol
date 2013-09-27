/*************************************************************************
	> File Name: LogisticLoss.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2013/8/18 星期日 17:11:42
	> Functions: Logistic loss for binary classification
 ************************************************************************/

#pragma once
#include "LossFunction.h"

namespace SOL
{
	template <typename FeatType, typename LabelType>
	class LogisticLoss: public LossFunction<FeatType, LabelType>
	{
		public:
			virtual double GetLoss(const DataPoint<FeatType, LabelType> &x, double predict)
			{
				double tmp = -predict * x.label;
				if (tmp > 100)
					return tmp; 
				else if (tmp < -100)
					return 0;
				else
					return std::log(1.0 + std::exp(tmp));
			}

            //aggressive learning 
			virtual double GetGradient(const DataPoint<FeatType, LabelType> &x, double predict)
			{
				double tmp = predict * x.label;
				if (tmp > 100) //to reject numeric problems
					return 0;
				else if (tmp  < -100)
					return -x.label; 
				else
					return -x.label / (1 + std::exp(tmp)); 
			}
	};
}
