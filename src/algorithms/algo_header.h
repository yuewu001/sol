#ifndef HEADER_ALGO_HEADER
#define HEADER_ALGO_HEADER

#include "om/olm/solm/solm_header.h"

namespace BOC{
    template <typename FeatType, typename LabelType>
        class ModelInfo{
		public:
            static void GetModelInfo(std::string & info){
                SOLModelInfo<FeatType, LabelType>::GetModelInfo(info);
            }
        };
}
#endif
