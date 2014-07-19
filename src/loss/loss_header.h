#ifndef HEADER_SOL_LOSS
#define HEADER_SOL_LOSS

#include "LogisticLoss.h"
#include "HingeLoss.h"
#include "SquareLoss.h"
#include "SquaredHingeLoss.h"
#include "MaxScoreHingeLoss.h"

namespace BOC{
	template <typename FeatType, typename LabelType>
	class LossInfo{
	public:
		static void GetLossInfo(std::string & info){
			info.append("\nLoss Functions:");
			APPEND_INFO(info, HingeLoss, FeatType, LabelType);
			APPEND_INFO(info, LogisticLoss, FeatType, LabelType);
			APPEND_INFO(info, SquareLoss, FeatType, LabelType);
			APPEND_INFO(info, SquaredHingeLoss, FeatType, LabelType);
			APPEND_INFO(info, MaxScoreHingeLoss, FeatType, LabelType);
		}
	};
}
#endif
