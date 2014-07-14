#ifndef HEADER_OL_MODEL_INFO
#define HEADER_OL_MODEL_INFO

#include "SGD.h"
#include "solm/solm_header.h"
#include "ofs/ofs_header.h"

#include <string>

namespace BOC{
    /**
     * @Synopsis Get all information of online learning algorithms
     *
     * @tparam FeatType
     * @tparam LabelType
     */
	template <typename FeatType, typename LabelType>
	class OLModelInfo{
	public:
		static void GetModelInfo(std::string & info){
			info.append("\nOnline Learning Algorithms:");
			APPEND_INFO(info,SGD,FeatType, LabelType);

            SOLModelInfo<FeatType,LabelType>::GetModelInfo(info);
            OFSModelInfo<FeatType, LabelType>::GetModelInfo(info);
		}
	};
}
#endif
