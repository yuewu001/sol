#ifndef HEADER_SOL_IO
#define HEADER_SOL_IO

#include "../utils/util.h"
#include "DataReader.h"
#include "libsvm_io.h"
#include "libsvm_binary.h"
#include "DataSet.h"
#include "MPDataSet.h"

namespace SOL{
	template <typename FeatType, typename LabelType>
	DataReader<FeatType, LabelType>* getReader(const string &filename, string dt_type){
		ToLowerCase(dt_type);
		if (dt_type == "libsvm")
			return new SOL::libsvm_io_<FeatType, LabelType>(filename);
		else if (dt_type == "cache")
			return new SOL::libsvm_binary_<FeatType,LabelType>(filename);
		else {
			std::cerr<<"unrecognized data reader type "<<dt_type<<std::endl;
			return NULL;
		}
	}

	template <typename FeatType, typename LabelType>
	DataSet<FeatType, LabelType>* getDataSet(int passNum, int buf_size, string dt_type, int mp_buf_size){
		ToLowerCase(dt_type);
		if (dt_type == "none")
			return new SOL::DataSet<FeatType, LabelType>(passNum, buf_size);
		else if (dt_type == "all")
			return new SOL::MPDataSet<FeatType, LabelType>(passNum, buf_size, MPBufferType_ALL,mp_buf_size);
		else if (dt_type == "margin")
			return new SOL::MPDataSet<FeatType, LabelType>(passNum, buf_size, MPBufferType_MARGIN,mp_buf_size);
		else if (dt_type == "reservior")
			return new SOL::MPDataSet<FeatType, LabelType>(passNum, buf_size, MPBufferType_RESERVIOR,mp_buf_size);
		else {
			std::cerr<<"ERROR: unrecognized dataset type "<<dt_type<<std::endl;
			return NULL;
		}
	}
}
#endif

