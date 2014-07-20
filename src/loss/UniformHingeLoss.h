/*************************************************************************
	> File Name: UniformHingeLoss.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2014/7/20 22:23:48
	> Functions: Uniform Hinge loss for multi-class
 ************************************************************************/

#ifndef HEADER_UNIFORM_HINGE_LOSS
#define HEADER_UNIFORM_HINGE_LOSS

#include "LossFunction.h"

#include <numeric>

namespace BOC {
	template <typename FeatType, typename LabelType>
	class UniformHingeLoss : public LossFunction < FeatType, LabelType > {

		//for dynamic binding
		DECLARE_CLASS

	public:
		UniformHingeLoss() :
			LossFunction<FeatType, LabelType>(MC_LOSS_TYPE){}

	public:
        virtual  void GetLoss(LabelType label, float* predict, float* loss, int len) {
            float tempLoss = 0;
            for (int i = 0; i < len; ++i){
                if (i == label)
                    continue;
                if (predict[label] < predict[i]){
                    tempLoss += predict[i];
                }
            }
            *loss = max(0.0f, 1.f - predict[label] + tempLoss);
        }

		virtual  void GetGradient(LabelType label, float *predict, float* gradient, float* classifier_weight, int len) {
            for (int i = 0; i < len; ++i){
                gradient[i] = 0;
                classifier_weight[i] = 0;
            }

            float tempLoss = 0;
            int tempLossId = -1;
            int errNum = 0;
            for (int i = 0; i < len; ++i){
                if (i != label && predict[label] <= predict[i]){
                    tempLoss += predict[i];
                    classifier_weight[i] = 1;
                    ++errNum;
                }
            }

            if (errNum > 0){
                for (int i = 0; i < len; ++i){
                    classifier_weight[i] /= errNum;
                }
                tempLoss /= errNum;
            }
            classifier_weight[label] = 1;

            tempLoss = max(0.0f, 1.f - predict[label] + tempLoss);

            if (tempLoss > 0){
                for (int i = 0; i < len; ++i) {
                    gradient[i] = classifier_weight[i];
                }
                gradient[label] = -1;
            }
        }
    };

    //for dynamic binding
    IMPLEMENT_LOSS_CLASS(UniformHingeLoss, UniformHinge)
}



#endif
