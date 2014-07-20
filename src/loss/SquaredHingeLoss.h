/*************************************************************************
	> File Name: SquaredHingeLoss.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2013/11/27 11:30:44
	> Functions: Squared Hinge loss
	************************************************************************/
#ifndef HEADER_SQUARE_HINGE_LOSS
#define HEADER_SQUARE_HINGE_LOSS

#include "LossFunction.h"

namespace BOC {
	template <typename FeatType, typename LabelType>
	class SquaredHingeLoss : public LossFunction<FeatType, LabelType> {
		//for dynamic binding
		DECLARE_CLASS

	public:
		SquaredHingeLoss() :
			LossFunction<FeatType, LabelType>(BC_LOSS_TYPE){}

	public:
		virtual  void GetLoss(LabelType label, float* predict, float* loss, int len) {
			*loss = max(0.0f, 1.f - *predict * label);
			*loss *= *loss;
		}

		virtual  void GetGradient(LabelType label, float* predict, float* gradient, float* classifier_weight,  int len) {
			float loss = max(0.0f, 1.f - *predict * label);
			if (loss > 0){
				*gradient = -label * loss * 2.f;
			}
			else{
				*gradient = 0;
			}
		}
	};

	//for dynamic binding
	IMPLEMENT_LOSS_CLASS(SquaredHingeLoss, SquaredHinge)
}

#endif
