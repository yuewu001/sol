/*************************************************************************
> File Name: LossFunction.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/18 Sunday 16:48:55
> Functions: base class for loss function
************************************************************************/

#ifndef HEADER_LOSS_FUNCTIONS
#define HEADER_LOSS_FUNCTIONS

#include "../utils/reflector.h"

namespace BOC {

#define IMPLEMENT_LOSS_CLASS(className, name) \
	template <typename FeatType, typename LabelType> \
	ClassInfo className<FeatType, LabelType>::classInfo(#name, "", className<FeatType, LabelType>::CreateObject); \
	\
	template <typename FeatType, typename LabelType> \
	void* className<FeatType, LabelType>::CreateObject(void *param1, void* param2, void* param3) \
	{ return new className<FeatType, LabelType>; }

	enum LossFunctionType{
		//loss function for binary classification
        BC_LOSS_TYPE = 0,
		//loss function for binary classification
		MC_LOSS_TYPE = 1,
	};

	template <typename FeatType, typename LabelType>
	class LossFunction : public Registry {

	public:
		inline static char Sign(float x) {
			if (x >= 0.f)
				return 1;
			else
				return -1;
		}

	protected:
		LossFunctionType lossType;

	public:
		LossFunction(LossFunctionType type) :
			lossType(type){ }

		LossFunctionType GetLossType() const { return this->lossType; }

	public:
		virtual void GetLoss(LabelType label, float* predict, float* loss, int len = 1) = 0;
		virtual void GetGradient(LabelType label, float* predict, float* gradient, float* classifier_weight = NULL, int len = 1) = 0;

	public:
		virtual ~LossFunction(){}
	};
}
#endif
