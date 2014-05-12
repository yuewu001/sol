#ifndef HEADER_SOL_ALGORITHMS
#define HEADER_SOL_ALGORITHMS

#include <vector>
#include <string>

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

static std::vector<std::string> algoNameList;

template <typename FeatType, typename LabelType>
void InitAlgorithms(){
	algoNameList.push_back(SGD<FeatType, LabelType>::classInfo.Type);
	algoNameList.push_back(Ada_FOBOS<FeatType, LabelType>::classInfo.Type);
	algoNameList.push_back(Ada_RDA<FeatType, LabelType>::classInfo.Type);
	algoNameList.push_back(CW_RDA<FeatType, LabelType>::classInfo.Type);
	algoNameList.push_back(CW_TG<FeatType, LabelType>::classInfo.Type);
	algoNameList.push_back(DAROW<FeatType, LabelType>::classInfo.Type);
	algoNameList.push_back(FOBOS<FeatType, LabelType>::classInfo.Type);
	algoNameList.push_back(FOFS<FeatType, LabelType>::classInfo.Type);
	algoNameList.push_back(PET<FeatType, LabelType>::classInfo.Type);
	algoNameList.push_back(RDA<FeatType, LabelType>::classInfo.Type);
	algoNameList.push_back(SOFS<FeatType, LabelType>::classInfo.Type);
	algoNameList.push_back(STG<FeatType, LabelType>::classInfo.Type);
}

#endif
