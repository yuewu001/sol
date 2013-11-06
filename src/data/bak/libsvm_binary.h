/*************************************************************************
  > File Name: libsvm_binary.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Sat 21 Sep 2013 10:52:41 PM SGT
  > Functions:  io for binary libsvm dataset
 ************************************************************************/

#pragma once

#include "DataReader.h"
#include "libsvm_binary_helper.h"

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
                FILE* file;

                //open mode, 0 is read ,1 is write
                int open_mode;

            private:
                z_stream strm_in;
                z_stream strm_out;

                //buffer
                unsigned char* buf_in_de;//decoded input buffer
                unsigned char* buf_in_en; //ecndoed input buffer, loaded from disk
                unsigned char* buf_out_de; //decoded output buffer
                unsigned char* buf_out_en; //encoded output buffer

                unsigned char* buf_in_pos; //current read buf position
                unsigned char* buf_out_pos; //current write buf position

                size_t buf_in_have; //current availabe bytes to read
                size_t buf_out_have;//current availabel bytes to write

                int write_flag; //Z_FINISH or Z_NO_FLUSH

            private:
                //thread safe
                MUTEX write_lock;
                CV buf_available;

                bool is_writing;
                bool is_bufferring;
#if WIN32
                HANDLE write_thread;
#else
                pthread_t write_thread;
#endif

            public:
                libsvm_binary_(const std::string &fileName) {
                    file = NULL;
                    this->fileName = fileName;

                    open_mode = -1;
                    this->buf_in_de = NULL;
                    this->buf_in_en = NULL;
                    this->buf_out_de = NULL;
                    this->buf_out_en = NULL;
                    this->is_bufferring = false;
                    this->is_writing = false;

                    //init thread-safety 
                    initialize_mutex(&this->write_lock);
                    initialize_condition_variable(&buf_available);
                }

                ~libsvm_binary_() {
                    this->Close();
                    this->release_buf();
                    delete_mutex(&write_lock);
                }

                void release_buf(){
                    if (this->buf_in_de != NULL)
                        delete []this->buf_in_de;
                    this->buf_in_de = NULL;
                    if (this->buf_in_en != NULL)
                        delete []this->buf_in_en;
                    this->buf_in_en = NULL;
                    if (this->buf_out_de != NULL)
                        delete []this->buf_out_de;
                    this->buf_out_de = NULL;
                    if (this->buf_out_en != NULL)
                        delete []this->buf_out_en;
                    this->buf_out_en = NULL;

                }
                //////////////////online mode//////////////////
            public:
                bool OpenReading() {
                    this->Close();
                    /* allocate deflate state */
                    this->strm_in.zalloc = Z_NULL;
                    this->strm_in.zfree = Z_NULL;
                    this->strm_in.opaque = Z_NULL;
                    if(inflateInit(&strm_in) != Z_OK)
                        return false;

                    file = fopen(fileName.c_str(), "rb");

                    if (file == NULL) {
                        printf("can't open input file %s\n",fileName.c_str());
                        return false;
                    }
                    open_mode = 0;
                    if (this->buf_in_de == NULL){
                        try{
                            this->buf_in_en = new unsigned char[ZLIB_BUF_SIZE];
                            this->buf_in_de = new unsigned char[ZLIB_BUF_SIZE];
                        }catch(std::bad_alloc &ex){
                            cerr<<"allocate memory for zlib input buffer failed!\n"<<ex.what()<<endl;
                            this->release_buf();
                            return false;
                        }
                    }

                    this->buf_in_pos = this->buf_in_de;
                    this->buf_in_have = 0;

                    return true;
                }
                bool OpenWriting() {
                    this->Close();
                    /* allocate inflate state */
                    this->strm_out.zalloc = Z_NULL;
                    this->strm_out.zfree = Z_NULL;
                    this->strm_out.opaque = Z_NULL;
                    this->strm_out.avail_in = 0;
                    this->strm_out.next_in = Z_NULL;
                    if(deflateInit(&strm_out,0) != Z_OK)
                        return false;

                    file = fopen(fileName.c_str(), "wb");
                    if (file == NULL) {
                        printf("can't open output file %s\n",fileName.c_str());
                        return false;
                    }
                    open_mode = 1;
                    if (this->buf_out_de == NULL){
                        try{
                            this->buf_out_de = new unsigned char[ZLIB_BUF_SIZE];
                            this->buf_out_en = new unsigned char[ZLIB_BUF_SIZE];
                        }catch(std::bad_alloc &ex){
                            cerr<<"allocate memory for zlib output buffer failed!\n"<<ex.what()<<endl;
                            this->release_buf();
                            return false;
                        }
                    }
                    this->buf_out_pos = this->buf_out_de;
                    this->buf_out_have = 0;

                    this->is_bufferring = false;
                    this->is_writing = false;
#if WIN32
                    write_hread = ::CreateThread(NULL, 0, static_cast<LPTHREAD_START_ROUTINE>(thread_write_cache<FeatType,LabelType>), this, NULL, NULL);
#else
                    pthread_create(&write_thread,NULL,thread_write_cache<FeatType,LabelType>,this);
#endif
                    return true;
                }

                void Rewind() {
                    if (file != NULL){
                        rewind(file);
                        if (this->open_mode == 0){
                            this->buf_in_pos = this->buf_in_de;
                            this->buf_in_have = 0;
                        }
                        else if (this->open_mode == 1){
                            this->buf_out_pos = this->buf_out_de;
                            this->buf_out_have = 0;

                        }
                    }
                }

                void Close() {
                    if (file != NULL) {
                        if (open_mode == 1){ //write
                            this->comp_data(this->buf_out_pos,0,Z_FINISH);
                            WaitThread(this->write_thread);

                            /* clean up and return */
                            (void)deflateEnd(&strm_out);

                        }
                        else{
                            /* clean up and return */
                            (void)inflateEnd(&strm_in);
                        }
                        fclose(file);
                        file = NULL;
                    }
                }

                inline bool Good() {
                    if (feof(file)){
                        //cout<<"reach end of file" <<endl;
                        return false;
                    }
                    else {
                        int errCode;
                        if (ferror(file) != 0){
                            this->Close();
                            ostringstream oss_errmsg;
                            oss_errmsg<<"unexpected file error occured at "<<__LINE__
                                <<" in file "<<__FILE__;
                            throw runtime_error(oss_errmsg.str().c_str());
                        }
                        else{
                            return true;
                        }
                    }
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
                    return true;
                }
                bool parse_data(unsigned char* dst, int len){
                    while (this->buf_in_have < len){
                        memcpy(dst, this->buf_in_pos, this->buf_in_have);
                        len -= this->buf_in_have;
                        dst += this->buf_in_have;
                        //this->buf_in_pos += this->buf_in_have; //can be ignored
                        this->buf_in_have = 0; //can be ignored

                        if (strm_in.avail_in == 0){
                            strm_in.avail_in = fread(this->buf_in_en, 1,ZLIB_BUF_SIZE,file);
                            if (ferror(file) || strm_in.avail_in == 0) {
                                (void)inflateEnd(&strm_in);
                                cerr<<"unexpected error occured when loading cache!"<<endl;
                                return false;
                            }
                            strm_in.next_in = this->buf_in_en;
                        }

                        /* run inflate() */
                        strm_in.avail_out = ZLIB_BUF_SIZE;
                        strm_in.next_out = this->buf_in_de; 
                        int ret = inflate(&strm_in, Z_NO_FLUSH);
                        assert(ret != Z_STREAM_ERROR);  /* state not clobbered */
                        switch (ret) {
                            case Z_NEED_DICT:
                                ret = Z_DATA_ERROR;     /* and fall through */
                            case Z_DATA_ERROR:
                            case Z_MEM_ERROR:
                                (void)inflateEnd(&strm_in);
                                cerr<<"error occured when parsing file!"<<endl;
                                return ret;
                        }
                        this->buf_in_have = ZLIB_BUF_SIZE - strm_in.avail_out;

                        this->buf_in_pos = this->buf_in_de;
                        if (this->buf_in_have == 0){
                            cerr<<"load compressed content failed!"<<endl;
                            return false;
                        }
                    }
                    memcpy(dst,this->buf_in_pos, len);
                    this->buf_in_pos += len;
                    this->buf_in_have -= len;
                    //len -= len; //can be ignored
                    //dst += len; //can be ignored

                    return true;
                }
                int comp_data(unsigned char* source, unsigned int len, int flush = Z_NO_FLUSH){
                    while(this->buf_out_have < len){
                        mutex_lock(&(this->write_lock));
                        this->write_flag = flush;
                        this->is_bufferring = true;

                        memcpy(this->buf_out_pos, source, this->buf_out_have);
                        source += this->buf_out_have;
                        len -= this->buf_out_have;
                        this->buf_out_have = 0; //can be ignored

                        this->is_bufferring = false;
                        if (this->is_writing == false){
                            //cout<<"parse thread: call for writing"<<endl;
                            condition_variable_signal_all(&(this->buf_available));
                        }

                        //cout<<"parse thread: wait"<<endl;
                        condition_variable_wait(&(this->buf_available), &(this->write_lock));
                        //cout<<"parse thread: continue to parse"<<endl;
                        mutex_unlock(&(this->write_lock));

                        this->buf_out_have = ZLIB_BUF_SIZE;
                        this->buf_out_pos = this->buf_out_de;
                    }

                    mutex_lock(&(this->write_lock));
                    this->write_flag = flush;
                    memcpy(this->buf_out_pos, source, len);
                    this->buf_out_pos += len;
                    this->buf_out_have -= len;
                    //len -= len; //can be ignored
                    //source += len; //can be ignored

                    if (flush == Z_FINISH){
                        this->is_bufferring = false;
                        if (this->is_writing == false){
                            //cout<<"parse thread: call for writing"<<endl;
                            condition_variable_signal_all(&(this->buf_available));
                        }
                        //cout<<"parse thread end: wait"<<endl;
                        condition_variable_wait(&(this->buf_available), &(this->write_lock));
                        //cout<<"parse thread end: continue to parse"<<endl;
                    }
                    mutex_unlock(&(this->write_lock));
                    return true;
                }
            private:
#if WIN32
                template <typename T1, typename T2> friend DWORD WINAPI thread_write_cache(LPVOID param);
#else
                template <typename T1, typename T2> friend void* thread_write_cache(void* param);
#endif

        };

    //for special definition
    typedef libsvm_binary_<float, char> libsvm_binary;
}
