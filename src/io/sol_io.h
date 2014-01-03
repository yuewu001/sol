#ifndef HEADER_SOL_IO
#define HEADER_SOL_IO

#include "DataReader.h"
#include "libsvm_io.h"
#include "libsvm_binary.h"

namespace SOL{
	template <typename FeatType, typename LabelType>
	DataReader<FeatType, LabelType>* getReader(const string &filename, const string& dataset_type){
		string dt_type = dataset_type;
		ToLowerCase(dt_type);
		if (dt_type == "libsvm")
			return new SOL::libsvm_io_<FeatType, LabelType>(filename);
		else if (dt_type == "cache")
			return new SOL::libsvm_binary_<FeatType,LabelType>(filename);
		else {
			std::cerr<<"unrecognized dataset type "<<dt_type<<std::endl;
			return NULL;
		}
	}
}
#endif

