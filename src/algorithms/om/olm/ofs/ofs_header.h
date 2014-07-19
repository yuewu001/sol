/*************************************************************************
	> File Name: ofs_header.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 7/11/2014 11:39:02 AM
	> Functions: header files for online feature selection
 ************************************************************************/
#ifndef HEADER_OFS_MODEL_INFO
#define HEADER_OFS_MODEL_INFO

#include "PET.h"
#include "FOFS.h"
//#include "SOFS.h"

#include <string>

namespace BOC{
    /**
     * @Synopsis Get all information of sparse online learning algorithms
     *
     * @tparam FeatType
     * @tparam LabelType
     */
	template <typename FeatType, typename LabelType>
	class OFSModelInfo{
	public:
		static void GetModelInfo(std::string & info){
			info.append("\nOnline Feature Selection  Algorithms:");
			APPEND_INFO(info,PET,FeatType, LabelType);
			APPEND_INFO(info,FOFS,FeatType, LabelType);
			//APPEND_INFO(info,SOFS,FeatType, LabelType);
		}
	};
}

#endif

