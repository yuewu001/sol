#ifndef HEADER_SOL_IO
#define HEADER_SOL_IO

#include "io_helper.h"

#include "OnlineDataSet.h"
#include "MPDataSet.h"

namespace SOL{
    template <typename FeatType, typename LabelType>
        DataSet<FeatType, LabelType>* getDataSet(int passNum, int buf_size, string dt_type, int mp_buf_size){
            ToLowerCase(dt_type);
            if (dt_type == "none")
                return new SOL::OnlineDataSet<FeatType, LabelType>(passNum, buf_size);
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

