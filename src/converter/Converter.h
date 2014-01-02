/*************************************************************************
  > File Name: Converter.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: 2014/1/2 10:26:58
  > Functions: Headers of Converter
 ************************************************************************/
#ifndef HEADER_CONVERTER
#define HEADER_CONVERTER

#include "Params.h"

#include "../io/DataReader.h"
#include "../io/libsvm_io.h"
#include "../io/libsvm_binary.h"
#include "../utils/util.h"

void Convert(const SOL::Params &param);
void Cache(const SOL::Params &param);

template <typename FeatType, typename LabelType>
SOL::DataReader<FeatType, LabelType>* getReader(const SOL::Params &param){
	string dt_type = param.str_data_type;
	ToLowerCase(dt_type);
	if (dt_type == "libsvm")
		return new SOL::libsvm_io_<FeatType, LabelType>(param.in_fileName);
	else if (dt_type == "cache")
		return new SOL::libsvm_binary_<FeatType,LabelType>(param.in_fileName);
	else {
		std::cerr<<"unrecognized dataset type "<<dt_type<<std::endl;
		return NULL;
	}
}

#endif
