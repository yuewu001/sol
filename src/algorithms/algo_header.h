#ifndef HEADER_ALGO_HEADER
#define HEADER_ALGO_HEADER

#include "om/olm/olm_header.h"

namespace BOC{
	template <typename FeatType, typename LabelType>
	class ModelInfo{
	public:
		static void GetModelInfo(std::string & info){
			OLModelInfo<FeatType, LabelType>::GetModelInfo(info);
		}
	};
}
#endif
