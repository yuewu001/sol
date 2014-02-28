/*************************************************************************
  > File Name: io_helper.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Fri 28 Feb 2014 08:12:54 PM SGT
  > Descriptions: 
 ************************************************************************/
#ifndef HEADER_IO_HELPER
#define HEADER_IO_HELPER

#include "../utils/util.h"
#include "DataHandler.h"
#include "libsvm_io.h"
#include "csv_io.h"
#include "libsvm_binary.h"

namespace SOL{
    template <typename FeatType, typename LabelType>
        DataReader<FeatType, LabelType>* getReader(const string &filename, string dt_type){
            ToLowerCase(dt_type);
            if (dt_type == "libsvm")
                return new SOL::libsvm_io_<FeatType, LabelType>(filename);
            else if (dt_type == "csv")
                return new SOL::csv_io_<FeatType,LabelType>(filename);
            else if (dt_type == "cache")
                return new SOL::libsvm_binary_<FeatType,LabelType>(filename);
            else {
                std::cerr<<"unrecognized data reader type "<<dt_type<<std::endl;
                return NULL;
            }
        }
    template <typename FeatType, typename LabelType>
        DataHandler<FeatType, LabelType>* getWriter(const string &filename, string dt_type){
            ToLowerCase(dt_type);
            if (dt_type == "libsvm")
                return new SOL::libsvm_io_<FeatType, LabelType>(filename);
            else if (dt_type == "csv")
                return new SOL::csv_io_<FeatType,LabelType>(filename);
            else if (dt_type == "cache")
                return new SOL::libsvm_binary_<FeatType,LabelType>(filename);
            else {
                std::cerr<<"unrecognized data reader type "<<dt_type<<std::endl;
                return NULL;
            }
        }

}

#endif
