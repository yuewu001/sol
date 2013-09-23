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
#include "../util.h"

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
    template <typename T1, typename T2> class DataSet;
    template <typename T1, typename T2> void* thread_LoadCache(void* param)
    {
        DataSet<T1,T2>* dataset = static_cast<DataSet<T1,T2>*>(param);
        libsvm_binary reader(dataset->cache_fileName);
        if (reader.OpenReading() == true)
        {
            dataset->ClearBuffer();
            for (size_t pass= 0; pass < dataset->passNum; pass++)
            {
                reader.Rewind();
                while(reader.Good())
                {
                    DataPoint<T1,T2> &data = dataset->wt_ptr->data; 
                    data.erase();

                    if (reader.GetNextData(data) == true)
                    {
                        mutex_lock(&dataset->data_lock); 
                        //notice that there is data available
                        dataset->wt_ptr = dataset->wt_ptr->next;
                        dataset->curDataNum++;
                        dataset->dataNum++;
                        condition_variable_signal_all(&dataset->data_available);

                        if (dataset->curDataNum == dataset->bufSize) //buffer full
                            condition_variable_wait(&dataset->buffer_full,&dataset->data_lock);
                        mutex_unlock(&dataset->data_lock);
                    }
                    else
                        break;
                }
            }
            //notice that the all the data has been loaded
            mutex_lock(&dataset->data_lock);
            dataset->load_finished = true;
            condition_variable_signal_all(&dataset->data_available);
            mutex_unlock(&dataset->data_lock);

            return NULL;
        }
        return NULL;
    }

    //data set, can work in both read-and-write mode and read-once mode
    template <typename FeatType, typename LabelType> class DataSet
    {		
        private:
            struct DataItem
            {
                DataPoint<FeatType, LabelType> data;
                DataItem *next;
            };

            string cache_fileName;

            size_t bufSize; //buffer to load data
            size_t passNum; //number of passes
            size_t dataNum; //total data number
        public:
            size_t curDataNum;  //data number in buffer

            //pointer to the first element, circlar linked list will be used
            DataItem *head; 
            DataItem *wt_ptr; //pointer to the write location
            DataItem *rd_ptr; //pointer to the read location

            bool load_finished;
            //thread-safety
            MUTEX data_lock;
            CV data_available;
            CV buffer_full;

        public:
            DataSet(size_t passes = 1, size_t buf_size = 128)
            {
                this->head = NULL;
                this->wt_ptr = NULL;
                this->rd_ptr = NULL;

                this->passNum = passes; 
                this->dataNum = 0;
                this->curDataNum = 0;

                this->load_finished = false;

                this->CreateBuffer(buf_size);

                //init thread-safety 
                initialize_mutex(&this->data_lock);
                initialize_condition_variable(&data_available);
                initialize_condition_variable(&buffer_full);
            }
            ~DataSet()
            {
                delete_mutex(&data_lock);
            }

        private:
            bool CreateBuffer(size_t buf_size = 128)
            {
                this->ReleaseBuffer();
                this->bufSize =  buf_size;
                if (this->bufSize <= 0)
                    return true;

                this->head = new DataItem;
                DataItem *p = this->head;
                for (size_t i = 1; i < this->bufSize; i++)
                {
                    p->next = new DataItem;
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
                DataItem *p = this->head;
                if (p == NULL)
                    return;
                p = p->next;
                while (p != this->head)
                {
                    p->data.erase();
                    p = p->next;
                }
                p->data.erase();
                this->dataNum = 0;
                this->curDataNum = 0;
                this->wt_ptr = this->head;
                this->rd_ptr = this->head;

            }
            void ReleaseBuffer()
            {
                DataItem *p = this->head;
                if (p == NULL)
                    return;
                DataItem *q = p->next;
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

            void LoadCache(const string& fileName)
            {
                this->cache_fileName = fileName;

                pthread_t thread1;
                pthread_create(&thread1,NULL,thread_LoadCache<FeatType,LabelType>,this);
            }

            //bind a data reader to the dataset
            bool Load(DataReader<FeatType, LabelType> &reader, const string &cache_file)
            {
                libsvm_binary writer(cache_file);
                if (writer.OpenWriting() == false)
                {
                    return false;
                }

                if (reader.OpenReading() == true)
                {
                    this->dataNum = 0;
                    DataPoint<FeatType, LabelType> data;
                    for (size_t pass= 0; pass < this->passNum; pass++)
                    {
                        reader.Rewind();
                        while(true)
                        {
                            data.erase();
                            if (reader.GetNextData(data) == true)
                            {
                                this->dataNum++;
                                writer.WriteData(data);
                            }
                            else
                                break;
                        }
                    }
                    return true;
                }
                else
                {
                    cerr<<"Can not open file to read!"<<endl;
                    return false;
                }
            }

            /////////////Data Access/////////////////////
        public:
            //get the data to read
            inline const DataPoint<FeatType, LabelType>& GetDataRd()
            {
                mutex_lock(&this->data_lock);
                
                //check if there is available data
                if (this->curDataNum <= 0) //no available data
                {
                    //suspend the current thread
                    if (this->load_finished == false)
                    {
                        condition_variable_wait(&this->data_available,&this->data_lock);
                        mutex_unlock(&this->data_lock);
                        return this->GetDataRd();
                    }
                    else
                    {
                        this->rd_ptr->data.erase();
                        mutex_unlock(&this->data_lock);
                        return this->rd_ptr->data; //return an invalid data
                    }
                }
                DataPoint<FeatType,LabelType>&data = this->rd_ptr->data;
                //notice to conitnue to read
                mutex_unlock(&this->data_lock);

                return data;
            }
            void FinishRead()
            {
                mutex_lock(&this->data_lock);
                this->rd_ptr = this->rd_ptr->next;
                this->curDataNum--;
                //notice that the last data have been processed
                //condition_variable_signal(&this->data_inuse);
                condition_variable_signal_all(&this->buffer_full);
                mutex_unlock(&this->data_lock);
            }

            //the number of features
            inline size_t size() const {return this->dataNum; }
    };

}
