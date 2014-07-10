#ifndef HEADER_ALGO_HEADER
#define HEADER_ALGO_HEADER

#include "om/om_header.h"

namespace BOC{
	template <typename FeatType, typename LabelType>
	class ModelInfo{
	public:
		static void GetModelInfo(std::string & info){
			OModelInfo<FeatType, LabelType>::GetModelInfo(info);
		}
	};
}
#endif
