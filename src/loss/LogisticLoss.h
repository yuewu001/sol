/*************************************************************************
	> File Name: LogisticLoss.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2013/8/18 Sunday 17:11:42
	> Functions: Logistic loss for binary classification
	************************************************************************/

#ifndef HEADER_LOGISTIC_FUNCTIONS
#define HEADER_LOGISTIC_FUNCTIONS
#include "LossFunction.h"

namespace BOC {
	template <typename FeatType, typename LabelType>
	class LogisticLoss : public LossFunction<FeatType, LabelType> {
		//for dynamic binding
		DECLARE_CLASS

	public:
		virtual bool IsCorrect(LabelType label, float* predict){
			return Sign(*predict) == label ? true : false;
		}

		virtual void GetLoss(LabelType label, float *predict, float* loss) {
			float tmp = -*predict * label;
			if (tmp > 100.f){
				*loss = tmp;
			}
			else if (tmp < -100.f){
				*loss = 0.f;
			}
			else{
				*loss = log(1.f + exp(tmp));
			}
		}

		//aggressive learning 
		virtual void GetGradient(LabelType label, float* predict, float* gradient) {
			float tmp = *predict * label;
			//to reject numeric problems
			if (tmp > 100.f) {
				*gradient = 0.f;
			}
			else if (tmp < -100.f){
				*gradient = -(float)(label);
			}
			else{
				*gradient = -label / (1.f + exp(tmp));
			}
		}
	};
	//for dynamic binding
	IMPLEMENT_LOSS_CLASS(LogisticLoss, logistic)
}

#endif
