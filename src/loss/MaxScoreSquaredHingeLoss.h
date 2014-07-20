/*************************************************************************
	> File Name: MaxScoreSquaredHingeLoss.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2014/7/20 3:43:28
	> Functions: Max Score Squared Hinge loss for multi-class
 ************************************************************************/
#ifndef HEADER_MAX_SCORE_SQUARED_HINGE_LOSS
#define HEADER_MAX_SCORE_SQUARED_HINGE_LOSS

#include "LossFunction.h"

#include <numeric>

namespace BOC {
	template <typename FeatType, typename LabelType>
	class MaxScoreSquaredHingeLoss : public LossFunction < FeatType, LabelType > {

		//for dynamic binding
		DECLARE_CLASS

	public:
		MaxScoreSquaredHingeLoss() :
			LossFunction<FeatType, LabelType>(MC_LOSS_TYPE){}

	public:
        virtual  void GetLoss(LabelType label, float* predict, float* loss, int len) {
            float tempLoss = -(std::numeric_limits<float>::max)();
            for (int i = 0; i < len; ++i){
                if (i == label)
                    continue;
                if (tempLoss < predict[i]){
                    tempLoss = predict[i];
                }
            }
            *loss = max(0.0f, 1.f - predict[label] + tempLoss);
            (*loss) *= (*loss);
        }

        virtual  void GetGradient(LabelType label, float *predict, float* gradient, float* classifier_weight, int len) {
            float tempLoss = -(std::numeric_limits<float>::max)();
			int tempLossId = -1;
            for (int i = 0; i < len; ++i){
                if (i == label)
                    continue;
                if (tempLoss < predict[i]){
                    tempLoss = predict[i];
					tempLossId = i;
                }
            }
            tempLoss = max(0.0f, 1.f - predict[label] + tempLoss);

			for (int i = 0; i < len; ++i){
				gradient[i] = 0;
				classifier_weight[i] = 0;
			}
			if (tempLoss > 0){
				gradient[tempLossId] = 2.f * tempLoss;
				gradient[label] = -2.f * tempLoss;
				classifier_weight[tempLossId] = 1;
				classifier_weight[label] = 1;
			}
        }
	};

	//for dynamic binding
    IMPLEMENT_LOSS_CLASS(MaxScoreSquaredHingeLoss, MaxScoreSquaredHinge)
}


#endif
