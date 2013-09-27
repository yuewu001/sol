/*************************************************************************
  > File Name: DataSet.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: 2013/8/18 星期日 15:38:09
  > Functions: Class to interact with datasets
 ************************************************************************/

#pragma once

#include "DataPoint.h"
#include "DataReader.h"
#include "libsvm_binary.h"
#include "libsvmread.h"
#include "../common/util.h"
#include "../common/global.h"

#include "thread_primitive.h"

#include <vector>
#include <string>
#include <fstream>

using namespace std;

/**
 *  namespace: Sparse Online Learning
 */
namespace SOL
{
    static const int init_buf_size = 2;

    template <typename T1, typename T2> class DataSet;
    template <typename T1, typename T2> void* thread_LoadCache(void* param)
    {
        DataSet<T1,T2>* dataset = static_cast<DataSet<T1,T2>*>(param);
        libsvm_binary_<T1,T2>* reader = dataset->reader;

        if (reader->Good())
        {
            for (size_t pass= 0; pass < dataset->passNum; pass++)
            {
                reader->Rewind();
                bool not_file_end = false;
                do
                {
                    DataChunk<T1,T2> &chunk = *dataset->wt_ptr;
                    chunk.erase();
                    not_file_end = reader->GetNextData(chunk);

                    mutex_lock(&dataset->data_lock); 
                    //notice that there is data available
                    dataset->wt_ptr = dataset->wt_ptr->next;
                    dataset->curChunkNum++; 
                    dataset->dataNum += chunk.dataNum;
                    condition_variable_signal_all(&dataset->data_available);

                    if (dataset->curChunkNum == dataset->bufSize) //buffer full
                    {
                        condition_variable_wait(&dataset->buffer_full,&dataset->data_lock);
                    }
                    mutex_unlock(&dataset->data_lock);

                }while(not_file_end == true);
            }

            //notice that the all the data has been loaded
            mutex_lock(&dataset->data_lock);
            dataset->load_finished = true;
            dataset->is_on_loading = false;
            condition_variable_signal_all(&dataset->data_available);
            cout<<"Load data finished!"<<endl;
            mutex_unlock(&dataset->data_lock);

            return NULL;
        }
        else
        {
            cout<<"reader is incorrect!"<<endl;
        }
        return NULL;
    }

    //data set, can work in both read-and-write mode and read-once mode
    template <typename FeatType, typename LabelType> class DataSet
    {		
        private:
            string cache_fileName;
            enum_DataSet_Type dataset_type;

            size_t bufSize; //buffer to load data
            size_t passNum; //number of passes
            size_t dataNum; //total data number

            size_t curChunkNum;  //data number in buffer

            //pointer to the first element, circlar linked list will be used
            DataChunk<FeatType,LabelType> *head; 
            DataChunk<FeatType,LabelType> *wt_ptr; //pointer to the write location
            DataChunk<FeatType,LabelType> *rd_ptr; //pointer to the read location

            bool load_finished; //this is used for GetChunk to test if current loading has finished
            bool is_on_loading; //this is used for Rewind to test if rewind can be performed

            libsvm_binary_<FeatType,LabelType> *reader;

            //thread-safety
            MUTEX data_lock;
            CV data_available;
            CV buffer_full;

        public:
            DataSet(size_t passes = 1, int buf_size = -1)
            {
                this->dataset_type = DataSet_Type_BC;
                this->head = NULL;
                this->wt_ptr = NULL;
                this->rd_ptr = NULL;

                this->passNum = passes; 
                this->dataNum = 0;
                this->curChunkNum = 0;

                this->load_finished = false;
                this->is_on_loading = false;
                this->reader = NULL;

                this->CreateBuffer(buf_size);

                //init thread-safety 
                initialize_mutex(&this->data_lock);
                initialize_condition_variable(&data_available);
                initialize_condition_variable(&buffer_full);
            }
            ~DataSet()
            {
                delete_mutex(&data_lock);
                if (this->reader != NULL)
                    delete this->reader;
            }

        private:
            bool CreateBuffer(int buf_size = 0)
            {
                this->bufSize = buf_size > 0 ? buf_size : init_buf_size;
                this->ReleaseBuffer();
                if (this->bufSize <= 0)
                    return true;

                this->head = new DataChunk<FeatType,LabelType>;
                DataChunk<FeatType,LabelType> *p = this->head;
                for (size_t i = 1; i < this->bufSize; i++)
                {
                    p->next = new DataChunk<FeatType,LabelType>;
                    p = p->next;
                }
                p->next = this->head;
                this->wt_ptr = this->head;
                this->rd_ptr = this->head;

                return true;
            }

        private:
            void ClearBuffer()
            {
                DataChunk<FeatType,LabelType> *p = this->head;
                if (p == NULL)
                    return;
                p = p->next;
                while (p != this->head)
                {
                    p->erase();
                    p = p->next;
                }
                p->erase();
                this->dataNum = 0;
                this->curChunkNum = 0;
                this->wt_ptr = this->head;
                this->rd_ptr = this->head;
            }

            void ReleaseBuffer()
            {
                DataChunk<FeatType,LabelType> *p = this->head;
                if (p == NULL)
                    return;
                DataChunk<FeatType,LabelType> *q = p->next;
                while (q != this->head)
                {
                    p = q->next;
                    delete q;
                    q = p;
                }
                delete this->head;
                this->head = NULL;
                this->wt_ptr = NULL;
                this->rd_ptr = NULL;
                this->dataNum = 0;
            } 

        public:
            template <typename T1, typename T2> friend void* thread_LoadCache(void* param);

            //bind a data reader to the dataset
            bool Load(const string &fileName)
            {
                this->cache_fileName = fileName;

                if (this->reader != NULL)
                    delete this->reader;
                this->reader = new libsvm_binary_<FeatType, LabelType>(this->cache_fileName);
                if (this->reader->OpenReading() == false)
                    return false;

                return true;
            }

            /////////////Data Access/////////////////////
        public:
            //get the data to read
            inline const DataChunk<FeatType, LabelType>& GetChunk()
            {
                mutex_lock(&this->data_lock);

                //check if there is available data
                if (this->curChunkNum <= 0) //no available data
                {
                    //suspend the current thread
                    if (this->load_finished == false)
                    {
                        condition_variable_wait(&this->data_available,&this->data_lock);
                        mutex_unlock(&this->data_lock);
                        return this->GetChunk();
                    }
                    else
                    {
                        this->rd_ptr->erase();
                        mutex_unlock(&this->data_lock);
                        return *this->rd_ptr; //return an invalid data
                    }
                }
                DataChunk<FeatType,LabelType> &chunk = *this->rd_ptr;
                //notice to conitnue to read
                mutex_unlock(&this->data_lock);

                return chunk;
            }

            void FinishRead()
            {
                mutex_lock(&this->data_lock);
                this->rd_ptr = this->rd_ptr->next;
                this->curChunkNum--;
                //notice that the last data have been processed
                condition_variable_signal_all(&this->buffer_full);
                mutex_unlock(&this->data_lock);
            }

            //the number of features
            inline size_t size() const {return this->dataNum; }
            bool Rewind()
            {
                mutex_lock(&this->data_lock);
                if (this->is_on_loading == true)
                {
                    cout<<"data is on loading"<<endl;
                    mutex_unlock(&this->data_lock);
                    return false;
                }
                reader->Rewind();
                this->ClearBuffer();
                this->load_finished = false;
                this->is_on_loading = true;
                mutex_unlock(&this->data_lock);

                pthread_t thread1;
                pthread_create(&thread1,NULL,thread_LoadCache<FeatType,LabelType>,this);
                return true;
            }
    };
}
