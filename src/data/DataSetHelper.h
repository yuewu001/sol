/*************************************************************************
  > File Name: DataSetHelper.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Thu 24 Oct 2013 03:33:10 PM
  > Descriptions: thread function definitions
 ************************************************************************/
#pragma once

#include "libsvm_binary.h"
#include "thread_primitive.h"

namespace SOL{
    template <typename T1, typename T2> class DataSet;
    template <typename T1, typename T2> 
#if WIN32
        DWORD WINAPI thread_LoadData(LPVOID param)
#else
        void* thread_LoadData(void* param)
#endif
        {
            DataSet<T1,T2>* dataset = static_cast<DataSet<T1,T2>*>(param);
            DataReader<T1,T2>* reader = dataset->reader;
            libsvm_binary_<T1,T2>* writer = NULL;

            string tmpFileName= dataset->cache_fileName + ".writing";

            if (dataset->is_cache == true){
                writer = new libsvm_binary_<T1,T2>(tmpFileName);
                if (writer->OpenWriting() == false){
                    cerr<<"Open cache file failed!"<<endl;
                    exit(0);
                }
            }

            //load data
            reader->Rewind();
            if (reader->Good()) {
                bool not_file_end = false;
                do {
                    DataChunk<T1,T2> &chunk = *dataset->wt_ptr;
                    chunk.erase();
                    for (size_t i = 0; i < init_chunk_size; i++) {
                        DataPoint<T1,T2> &data = chunk.data[i];
                        not_file_end = reader->GetNextData(data);
                        if (not_file_end == true){
                            chunk.dataNum++;
                            if (writer != NULL){
                                writer->WriteData(data);
                            }
                        }
                        else{
                            break;
                        }
                    }

					if(chunk.dataNum > 0){
						mutex_lock(&dataset->data_lock); 
						//notice that there is data available
						dataset->wt_ptr = dataset->wt_ptr->next;
						dataset->curChunkNum++; 
						dataset->dataNum += chunk.dataNum;
						condition_variable_signal_all(&dataset->data_available);

						if (dataset->curChunkNum == dataset->bufSize){ //buffer full
							condition_variable_wait(&dataset->buffer_full,&dataset->data_lock);
						}
						mutex_unlock(&dataset->data_lock);
					}

                }while(not_file_end == true);
            }
            else{
                cerr<<"load data failed!"<<endl;
                exit(0);
            }

            if (dataset->is_cache == true){
                dataset->reader->Close();
                delete dataset->reader;

                writer->Close();
                delete writer;

                //rename
#if WIN32
                string cmd = "REN \"";
                cmd = cmd + tmpFileName + "\" \"";
                cmd = cmd + dataset->cache_fileName + "\"";
#else
                string cmd = "mv \"";
                cmd = cmd + tmpFileName + "\" \"";
                cmd = cmd + dataset->cache_fileName + "\"";
#endif
                if(system(cmd.c_str()) != 0){
                    cerr<<"rename cahe file name failed!"<<endl;
                    exit(0);
                }
                //load cache file
                dataset->reader = new libsvm_binary_<T1,T2>(dataset->cache_fileName);
                if (dataset->reader->OpenReading() == false){
                    cerr<<"load cache data failed!"<<endl;
                    exit(0);
                }
                reader = dataset->reader;
                dataset->is_cache = false;
            }

            //load cache
            for (size_t pass= 1; pass < dataset->passNum; pass++) {
                reader->Rewind();
                if (reader->Good()) {
                    bool not_file_end = false;
                    do {
                        DataChunk<T1,T2> &chunk = *dataset->wt_ptr;
                        chunk.erase();
                        for (size_t i = 0; i < init_chunk_size; i++) {
                            DataPoint<T1,T2> &data = chunk.data[i];
                            not_file_end = reader->GetNextData(data);
                            if (not_file_end == true)
                                chunk.dataNum++;
                            else
                                break;
                        }

						if (chunk.dataNum > 0){
							mutex_lock(&dataset->data_lock); 
							//notice that there is data available
							dataset->wt_ptr = dataset->wt_ptr->next;
							dataset->curChunkNum++; 
							dataset->dataNum += chunk.dataNum;
							condition_variable_signal_all(&dataset->data_available);

							if (dataset->curChunkNum == dataset->bufSize){ //buffer full
								condition_variable_wait(&dataset->buffer_full,&dataset->data_lock);
							}
							mutex_unlock(&dataset->data_lock);
						}

                    }while(not_file_end == true);
                }
                else {
                    cerr<<"reader is incorrect!"<<endl;
                    exit(0);
                }
            }

            //notice that the all the data has been loaded
            mutex_lock(&dataset->data_lock);
            dataset->load_finished = true;
            dataset->is_on_loading = false;
            condition_variable_signal_all(&dataset->data_available);
            mutex_unlock(&dataset->data_lock);

            return NULL;
        }
}
