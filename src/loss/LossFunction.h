/*************************************************************************
> File Name: LossFunction.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/18 Sunday 16:48:55
> Functions: base class for loss function
************************************************************************/

#ifndef HEADER_LOSS_FUNCTIONS
#define HEADER_LOSS_FUNCTIONS

namespace BOC {
	template <typename FeatType, typename LabelType>
	class LossFunction {
        inline char Sign(float x) {
            if (x >= 0.f) 
                return 1;
            else
                return -1;
        }

        public:
		virtual inline bool IsCorrect(LabelType label, float predict) {
            return Sign(predict) == label ? true : false;
        }

        virtual float GetLoss(LabelType label, float predict) = 0;
        virtual float GetGradient(LabelType label, float predict) = 0;

	public:
		virtual ~LossFunction(){}
    };
}
#endif
