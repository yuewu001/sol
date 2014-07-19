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

#define IMPLEMENT_LOSS_CLASS(className,name) \
	template <typename FeatType, typename LabelType> \
	ClassInfo className<FeatType, LabelType>::classInfo(#name, "", className<FeatType, LabelType>::CreateObject); \
	\
	template <typename FeatType, typename LabelType> \
	void* className<FeatType, LabelType>::CreateObject(void *param1, void* param2, void* param3) \
	{ return new className<FeatType, LabelType>; }

	template <typename FeatType, typename LabelType>
	class LossFunction : public Registry {
	protected:
		inline char Sign(float x) {
			if (x >= 0.f)
				return 1;
			else
				return -1;
		}

	public:
		virtual bool IsCorrect(LabelType label, float* predict) = 0;
		virtual void GetLoss(LabelType label, float* predict, float* loss) = 0;
		virtual void GetGradient(LabelType label, float* predict, float* gradient) = 0;

	public:
		virtual ~LossFunction(){}
	};
}
#endif
