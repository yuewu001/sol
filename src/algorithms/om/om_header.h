/*************************************************************************
	> File Name: om_header.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 7/10/2014 3:57:48 PM
	> Functions:
	************************************************************************/

#ifndef HEADER_ALGO_OM_HEADER
#define HEADER_ALGO_OM_HEADER

#include "olm/olm_header.h"

namespace BOC{
	template <typename FeatType, typename LabelType>
	class OModelInfo{
	public:
		static void GetModelInfo(std::string& info){
			OLModelInfo<FeatType, LabelType>::GetModelInfo(info);
		}
	};
}

#endif
