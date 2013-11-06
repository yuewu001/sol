/*************************************************************************
  > File Name: libsvm_binary_helper.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Tue 05 Nov 2013 09:29:15 PM
  > Descriptions: thread function for libsvm binary
 ************************************************************************/
#ifndef HEADER_LIBSVM_HEADER
#define HEADER_LIBSVM_HEADER

#include "thread_primitive.h"

#include <cstdio>
#include <string>
#include <assert.h>
#include "zlib.h"

namespace SOL{

#if defined(MSDOS) || defined(OS2) || defined(WIN32) || defined(__CYGWIN__)
#  include <fcntl.h>
#  include <io.h>
#  define SET_BINARY_MODE(file) setmode(fileno(file), O_BINARY)
#else
#  define SET_BINARY_MODE(file)
#endif

#define ZLIB_BUF_SIZE 4194304 //4 M

    template <typename T1, typename T2> class libsvm_binary_;
    template <typename T1, typename T2> 
#if WIN32
        DWORD WINAPI thread_write_cache(LPVOID param)
#else
        void* thread_write_cache(void* param)
#endif
        {
            libsvm_binary_<T1, T2>* writer = static_cast<libsvm_binary_<T1, T2>*>(param);
            if (write == NULL) {
                cerr<<"incorrect input for write thread!"<<endl;
                exit(1);
            }
            //buf
            unsigned char* buffer = NULL;
            try{
                buffer = new unsigned char[ZLIB_BUF_SIZE];
            }catch(std::bad_alloc &ex){
                cerr<<"allocate memory for threaded buffer failed!"<<endl;
                return NULL;
            }
            int write_flag;

            size_t total_size = 0;
            while(1){
                mutex_lock(&(writer->write_lock));
                if (writer->is_bufferring == true){
                    //cout<<"write thread: sleep for writing"<<endl;
                    writer->is_writing = false;
                    condition_variable_wait(&(writer->buf_available), &(writer->write_lock));
                    //cout<<"write thread: continue to write"<<endl;
                    writer->is_writing = true;
                }

                //copy data
                write_flag = writer->write_flag;
                writer->strm_out.avail_in = ZLIB_BUF_SIZE - writer->buf_out_have;
                total_size += writer->strm_out.avail_in;
                //cout<<"compress "<<writer->strm_out.avail_in<<" bytes!\n";
                memcpy(buffer, writer->buf_out_de,writer->strm_out.avail_in);
                writer->strm_out.next_in = buffer;

                writer->is_bufferring = true;

                //cout<<"write thread: call parse thread to parse"<<endl;
                condition_variable_signal_all(&(writer->buf_available));
                mutex_unlock(&(writer->write_lock));

                // run deflate()
                do {
                    writer->strm_out.avail_out = ZLIB_BUF_SIZE;
                    writer->strm_out.next_out = writer->buf_out_en;
                    int ret = deflate(&(writer->strm_out), write_flag);   //no bad return value 
                    assert(ret != Z_STREAM_ERROR);  // state not clobbered 
                    unsigned int have = ZLIB_BUF_SIZE - writer->strm_out.avail_out;
                    //cout<<"write "<<have<<" bytes!\n";
                    if (fwrite(writer->buf_out_en, 1, have,writer->file) != have || ferror(writer->file)) {
                        (void)deflateEnd(&(writer->strm_out));
                        cerr<<"unexpected error occured when writing file!"<<endl;
                        exit(1);
                    }
                } while (writer->strm_out.avail_out == 0);
                assert(writer->strm_out.avail_in == 0);     // all input will be used 

                if (write_flag == Z_FINISH){
                    break;
                }
            }

            if (buffer != NULL)
                delete []buffer;
            mutex_lock(&(writer->write_lock));
            writer->is_writing = false;
            mutex_unlock(&(writer->write_lock));
            //cout<<"total size: "<<total_size<<" bytes "<<endl;
            return NULL;
        }
}
#endif

