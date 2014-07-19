/*************************************************************************
	> File Name: HingeLoss.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2013/8/18 Sunday 16:58:22
	> Functions: Hinge Loss function, for SVM
	************************************************************************/

#ifndef HEADER_HINGE_LOSS
#define HEADER_HINGE_LOSS

#include "LossFunction.h"

namespace BOC {
	template <typename FeatType, typename LabelType>
	class HingeLoss : public LossFunction < FeatType, LabelType > {
		//for dynamic binding
		DECLARE_CLASS

	public:
		HingeLoss() :
			LossFunction<FeatType, LabelType>(BC_LOSS_TYPE){}

	public:
		virtual  void GetLoss(LabelType label, float* predict, float* loss, int len) {
			*loss = max(0.0f, 1.f - *predict * label);
		}

		virtual  void GetGradient(LabelType label, float *predict, float* gradient, int len) {
			float loss = 0;
			this->GetLoss(label, predict, &loss, len);
			if (loss > 0){
				*gradient = (float)(-label);
			}
			else{
				*gradient = 0;
			}
		}
	};

	//for dynamic binding
	IMPLEMENT_LOSS_CLASS(HingeLoss, hinge)
}

#endif
