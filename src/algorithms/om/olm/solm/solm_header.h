#ifndef HEADER_SOL_MODEL_INFO
#define HEADER_SOL_MODEL_INFO

#include "SGD.h"
#include "Ada_FOBOS.h"
#include "Ada_RDA.h"
#include "CW_RDA.h"
#include "CW_TG.h"
#include "DAROW.h"
#include "FOBOS.h"
#include "FOFS.h"
#include "PET.h"
#include "RDA.h"
#include "SOFS.h"
#include "STG.h"

//#include "mRMR_OGD.h"
#include <string>

namespace BOC{
    template <typename FeatType, typename LabelType>
        class SOLModelInfo{
            public:
                static void GetModelInfo(std::string & info){
                    info.append("Sparse Online Learning Algorithms:");
					info.append("\n\t"); 
					APPEND_INFO(info, SGD, FeatType, LabelType);
					//info.append(SGD<FeatType, LabelType>::classInfo.GetType());
                    //APPEND_INFO(info, SGD,FeatType, LabelType);
                    //algoList.push_back(Ada_FOBOS<FeatType, LabelType>::classInfo.Type);
                    //algoList.push_back(Ada_RDA<FeatType, LabelType>::classInfo.Type);
                    //algoList.push_back(CW_RDA<FeatType, LabelType>::classInfo.Type);
                    //algoList.push_back(CW_TG<FeatType, LabelType>::classInfo.Type);
                    //algoList.push_back(DAROW<FeatType, LabelType>::classInfo.Type);
                    //algoList.push_back(FOBOS<FeatType, LabelType>::classInfo.Type);
                    //algoList.push_back(FOFS<FeatType, LabelType>::classInfo.Type);
                    //algoList.push_back(PET<FeatType, LabelType>::classInfo.Type);
                    //algoList.push_back(RDA<FeatType, LabelType>::classInfo.Type);
                    //algoList.push_back(SOFS<FeatType, LabelType>::classInfo.Type);
                    //algoList.push_back(STG<FeatType, LabelType>::classInfo.Type);
                }
        };
}
#endif
