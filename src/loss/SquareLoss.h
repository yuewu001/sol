/*************************************************************************
	> File Name: SquareLoss.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2013/8/18 Sunday 17:19:33
	> Functions: Square Loss
	************************************************************************/

#ifndef HEADER_SQUARE_LOSS
#define HEADER_SQUARE_LOSS

#include "LossFunction.h"

namespace BOC {
	template <typename FeatType, typename LabelType>
	class SquareLoss : public LossFunction<FeatType, LabelType> {
		//for dynamic binding
		DECLARE_CLASS

	public:
		SquareLoss() :
			LossFunction<FeatType, LabelType>(BC_LOSS_TYPE){}


	public:
		virtual bool IsCorrect(LabelType label, float* predict, int len){
			return Sign(*predict) == label ? true : false;
		}

		virtual void GetLoss(LabelType label, float* predict, float* loss, int len) {
			*loss = (*predict - label) * (*predict - label);
		}

		virtual void GetGradient(LabelType label, float* predict, float* gradient, int len) {
			*gradient = 2 * (*predict - label);
		}
	};
	//for dynamic binding
	IMPLEMENT_LOSS_CLASS(SquareLoss, square)
}

#endif 
