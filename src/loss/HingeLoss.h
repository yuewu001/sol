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
		virtual bool IsCorrect(LabelType label, float* predict){
			return Sign(*predict) == label ? true : false;
		}

		virtual  void GetLoss(LabelType label, float* predict, float* loss) {
			*loss = max(0.0f, 1.f - *predict * label);
		}

		virtual  void GetGradient(LabelType label, float *predict, float* gradient) {
			float loss = 0;
			this->GetLoss(label, predict, &loss);
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
