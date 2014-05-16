/*************************************************************************
	> File Name: opt_helper.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 5/15/2014 11:13:44 PM
	> Functions: optimizer header
	************************************************************************/
#ifndef HEADER_OPTIMIZER_HELPER
#define HEADER_OPTIMIZER_HELPER

#include "../optimizers/OnlineOptimizer.h"
#include <string>

namespace BOC{
	template <typename FeatType, typename LabelType>
	class OptInfo{
	public:
		static void GetOptInfo(std::string & info){
			info.append("\nOptimizers:");
			APPEND_INFO(info, OnlineOptimizer, FeatType, LabelType);
		}
	};
}

#endif

