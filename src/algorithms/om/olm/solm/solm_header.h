#ifndef HEADER_SOL_MODEL_INFO
#define HEADER_SOL_MODEL_INFO

#include "Ada_FOBOS.h"
#include "Ada_RDA.h"
#include "CW_RDA.h"
#include "CW_TG.h"
#include "DAROW.h"
#include "FOBOS.h"
#include "RDA.h"
#include "STG.h"

//#include "mRMR_OGD.h"
#include <string>

namespace BOC{
    /**
     * @Synopsis Get all information of sparse online learning algorithms
     *
     * @tparam FeatType
     * @tparam LabelType
     */
	template <typename FeatType, typename LabelType>
	class SOLModelInfo{
	public:
		static void GetModelInfo(std::string & info){
			info.append("\nSparse Online Learning Algorithms:");
			APPEND_INFO(info,Ada_FOBOS, FeatType, LabelType);
			APPEND_INFO(info,Ada_RDA,FeatType, LabelType);
			APPEND_INFO(info,CW_RDA,FeatType, LabelType);
			APPEND_INFO(info,CW_TG,FeatType, LabelType);
			APPEND_INFO(info,DAROW,FeatType, LabelType);
			APPEND_INFO(info,FOBOS,FeatType, LabelType);
			APPEND_INFO(info,FOFS,FeatType, LabelType);
			APPEND_INFO(info,PET,FeatType, LabelType);
			APPEND_INFO(info,RDA,FeatType, LabelType);
			APPEND_INFO(info,SOFS,FeatType, LabelType);
			APPEND_INFO(info,STG,FeatType, LabelType);
		}
	};
}
#endif
