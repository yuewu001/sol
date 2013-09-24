/*************************************************************************
	> File Name: SquareLoss.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2013/8/18 星期日 17:19:33
	> Functions: Square Loss
 ************************************************************************/

#pragma once
#include "LossFunction.h"

namespace SOL
{
	template <typename FeatType, typename LabelType>
	class SquareLoss: public LossFunction<FeatType, LabelType>
	{
		
		public:
			virtual  bool IsCorrect(const DataPoint<FeatType, LabelType> &x, double predict)
			{				
				return predict == x.Label() ? true : false;
			} 

			virtual  double GetLoss(const DataPoint<FeatType, LabelType> &x, double predict)
			{
				return (predict - x.Label()) * (predict - x.Label());
			}

			virtual  double GetGradient(const DataPoint<FeatType, LabelType> &x, double predict, int index)
			{
				return 2 * (predict - x.Label()) * x[index];
			}

			virtual  double GetBiasGradient(const DataPoint<FeatType, LabelType> &x, double predict)
			{
				return 2 * (predict - x.Label()); 
			}
	};
}
