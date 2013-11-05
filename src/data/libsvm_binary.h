/*************************************************************************
  > File Name: libsvm_binary.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Sat 21 Sep 2013 10:52:41 PM SGT
  > Functions:  io for binary libsvm dataset
 ************************************************************************/

#pragma once

#include <cstdio>
#include <string>
#include <assert.h>
#include "zlib.h"


#include "DataReader.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <new>

using namespace std;

namespace SOL {
    template <typename FeatType, typename LabelType>
    class libsvm_binary_:public DataReader<FeatType, LabelType> {
        private:
            std::string fileName;
            gzFile file;

            //open mode, 0 is read ,1 is write
            int open_mode;

        public:
            libsvm_binary_(const std::string &fileName) {
                file = NULL;
                this->fileName = fileName;
                
                open_mode = -1;
            }
            ~libsvm_binary_() {
                this->Close();
            }

            //////////////////online mode//////////////////
        public:
            bool OpenReading() {
                this->Close();

                file = gzopen(fileName.c_str(), "rb");

                if (file == NULL) {
                    printf("can't open input file %s\n",fileName.c_str());
                    return false;
                }
                open_mode = 0;
                return true;
            }
            bool OpenWriting() {
                this->Close();
                file = gzopen(fileName.c_str(), "wb6");
                if (file == NULL) {
                    printf("can't open output file %s\n",fileName.c_str());
                    return false;
                }
                open_mode = 1;
                return true;
            }

            void Rewind() {
                if (file != NULL)
                    gzrewind(file);
            }
            void Close() {
                if (file != NULL) {
                    gzclose(file);
                    file = NULL;
                }
            }

            inline bool Good() {
                if (gzeof(file)){
                    //cout<<"reach end of file" <<endl;
                    return false;
                }
                else {
                    int errCode;
                    const char* errmsg = gzerror(file ,&errCode);;
                    if (errCode != Z_OK){
                        this->Close();
                        ostringstream oss_errmsg;
                        oss_errmsg<<errmsg<<__LINE__
                            <<" in file "<<__FILE__;
                        throw runtime_error(oss_errmsg.str().c_str());
                    }
                    else{
                        return true;
                    }
                }
            }

            bool parse_data(unsigned char* dst, int len){
                int cur_avail = gzread(file, dst, len);
                return cur_avail == len ? true : false;
            }

            bool GetNextData(DataPoint<FeatType, LabelType> &data) {
                data.erase();
                int featNum = 0;
                bool ret;
                ret = this->parse_data((unsigned char*)&(data.label), sizeof(LabelType));
                if (ret == true){
                    ret = this->parse_data((unsigned char*)&featNum, sizeof(int));
                    if (ret == true){
                        if(featNum > 0){
                            ret = this->parse_data((unsigned char*)&(data.max_index), sizeof(int));
                            if (ret == true){
                                data.indexes.resize(featNum);
                                ret = this->parse_data((unsigned char*)(data.indexes.begin), 
                                        sizeof(int) * featNum);
                            }
                            if (ret == true){
                                data.features.resize(featNum);
                                ret = this->parse_data((unsigned char*)(data.features.begin),
                                        sizeof(FeatType) * featNum);
                                return true;
                            }
                        }
                        else
                            return true;
                    }
                }
                else{
                    return Good();
                }
                cerr<<"error occured when parse compressed data!"<<endl;
                exit(1);
                return false;

                /*
                   file.read((char*)&data.label,sizeof(LabelType));
                   file.read((char*)&featNum,sizeof(int));
                   if (featNum > 0){
                   file.read((char*)&(data.max_index),sizeof(int));
                   data.indexes.resize(featNum);
                   file.read((char*)data.indexes.begin,sizeof(int) * featNum);
                   data.features.resize(featNum);
                   file.read((char*)data.features.begin,sizeof(FeatType) * featNum);
                //file.read((char*)data.weights.begin,sizeof(float) * featNum);
                }
                else
                return Good();
                return true;
                */
            }

            bool WriteData(DataPoint<FeatType, LabelType> &data) {
                int featNum = data.indexes.size();
                this->comp_data((unsigned char*)&data.label, sizeof(LabelType));
                this->comp_data((unsigned char*)&featNum, sizeof(int));
                if (featNum > 0){
                    this->comp_data((unsigned char*)&(data.max_index), sizeof(int));
                    this->comp_data((unsigned char*)(data.indexes.begin), sizeof(int) * featNum);
                    this->comp_data((unsigned char*)(data.features.begin), sizeof(FeatType) * featNum);
                }
                /*
                file.write((const char*)&data.label,sizeof(LabelType));
                file.write((const char*)&featNum,sizeof(int));
                if (featNum > 0){
                    file.write((const char*)&(data.max_index),sizeof(int));
                    file.write((const char*)data.indexes.begin,sizeof(int) * featNum);
                    file.write((const char*)data.features.begin,sizeof(FeatType) * featNum);
                }
                */
                //file.write((const char*)data.weights.begin,sizeof(float) * featNum);
                return true;
            }

            int comp_data(unsigned char* source, unsigned int len, int flush = Z_NO_FLUSH){
                int num_write = gzwrite(file, source, len);
                return (num_write > 0 ) ? num_write : 0;
            }
};

//for special definition
typedef libsvm_binary_<float, char> libsvm_binary;
}
