#ifndef HEADER_MODEL_INFO
#define HEADER_MODEL_INFO

#include "om/olm/solm/ModelInfo.h"

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
