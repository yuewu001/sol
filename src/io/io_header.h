/*************************************************************************
  > File Name: io_helper.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Fri 28 Feb 2014 08:12:54 PM SGT
  > Descriptions: io header
  ************************************************************************/
#ifndef HEADER_IO_HEADER
#define HEADER_IO_HEADER

#include "../utils/util.h"
#include "DataHandler.h"
#include "libsvm_io.h"
#include "csv_io.h"
#include "binary_io.h"

#include "OnlineDataSet.h"

#include <string>

namespace BOC{
	template <typename FeatType, typename LabelType>
	class IOInfo{
	public:
		static void GetIOInfo(std::string &info){
			info.append("\nDataset Readers:");
			APPEND_INFO(info, csv_io, FeatType, LabelType);
			APPEND_INFO(info, libsvm_io, FeatType, LabelType);
			APPEND_INFO(info, binary_io, FeatType, LabelType);
		}
	};
}

#endif
