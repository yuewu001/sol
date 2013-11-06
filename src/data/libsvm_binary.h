/*************************************************************************
  > File Name: libsvm_binary.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Sat 21 Sep 2013 10:52:41 PM SGT
  > Functions:  io for binary libsvm dataset
 ************************************************************************/

#ifndef HEADER_LIBSVM_BINARY
#define HEADER_LIBSVM_BINARY

#include "DataReader.h"

#include <new>


using namespace std;

namespace SOL {


    template <typename FeatType, typename LabelType>
        class libsvm_binary_:public DataReader<FeatType, LabelType> {
            private:
                std::string fileName;
                basic_io io_handler;

            public:
                libsvm_binary_(const std::string &fileName) {
                    this->fileName = fileName;
                }

                ~libsvm_binary_() {
                    this->Close();
                }
                
                //////////////////online mode//////////////////
            public:
                bool OpenReading() {
                    this->Close();
                    return io_handler.open_file(this->fileName.c_str(), "rb");
                }

                bool OpenWriting() {
                    this->Close();
                    return io_handler.open_file(this->fileName.c_str(), "wb");
                }

                void Rewind() {
                    io_handler.rewind();
                }

                void Close() {
                    io_handler.close_file();
                }

                inline bool Good() {
                    return io_handler.good() == 0 ? true : false;
                }

                bool GetNextData(DataPoint<FeatType, LabelType> &data) {
                    data.erase();
                    int featNum = 0;
                    size_t read_len = 0;
                    read_len = sizeof(LabelType);
                    if (io_handler.read_data((char*)&(data.label), read_len) == read_len){
                        read_len = sizeof(int);
                        if(io_handler.read_data((char*)&featNum, read_len) == read_len){
                            if(featNum > 0){
                                if(io_handler.read_data((char*)&(data.max_index), 
                                            read_len) == read_len){
                                    data.indexes.resize(featNum);
                                    read_len = sizeof(int) * featNum;
                                    if(io_handler.read_data((char*)(data.indexes.begin), 
                                            read_len) == read_len){
                                        data.features.resize(featNum);
                                        read_len = sizeof(float) * featNum;
                                        if(io_handler.read_data((char*)(data.features.begin),
                                            read_len) == read_len){
                                            return true;
                                        }
                                    }
                                }
                            }
                            else
                                return true;
                        }
                    }
                    else if (this->Good() == true){
                        return false;
                    }
                    else{
                        cerr<<"unexpected error occured when loading data!"<<endl;
                        exit(1);
                    }
                    return false;
                }

                bool WriteData(DataPoint<FeatType, LabelType> &data) {
                    int featNum = data.indexes.size();
                    size_t w_len = sizeof(LabelType);
                    if(io_handler.write_data((char*)&data.label, w_len) == w_len){
                        w_len = sizeof(int);
                        if(io_handler.write_data((char*)&featNum, w_len) == w_len){
                            if (featNum > 0){
                                if(io_handler.write_data((char*)&(data.max_index), 
                                            w_len) == w_len){
                                    w_len = sizeof(int) * featNum;
                                    if(io_handler.write_data((char*)(data.indexes.begin),
                                                w_len) == w_len){
                                        w_len = sizeof(float) * featNum;
                                        if(io_handler.write_data((char*)(data.features.begin), 
                                                    w_len) == w_len)
                                            return true;
                                    }
                                }
                            }
                            else{
                                return true;
                            }
                        }
                    }
                    return false;
                }
        };

    //for special definition
    typedef libsvm_binary_<float, char> libsvm_binary;
}

#endif
