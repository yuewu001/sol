/*************************************************************************
> File Name: DataSet.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/18 星期日 15:38:09
> Functions: Class to interact with datasets
************************************************************************/

#pragma once

#include "DataPoint.h"
#include "DataReader.h"
#include "../util.h"


#include <vector>
#include <string>
#include <fstream>

using namespace std;

/**
*  namespace: Sparse Online Learning
*/
namespace SOL
{
    //data set, can work in both read-and-write mode and read-once mode
	template <typename FeatType, typename LabelType> class DataSet
    {		
        struct DataItem
        {
            DataPoint<FeatType, LabelType> data;
            DataItem *next;
        };

        private:
            size_t bufSize; //buffer to load data
            size_t passNum; //number of passes
            size_t dataNum;


            //pointer to the first element, circlar linked list will be used
            DataItem *head; 
            DataItem *wt_ptr; //pointer to the write location
            DataItem *rd_ptr; //pointer to the read location

            s_array<size_t> indexMap; //index map, for random fatures

            //The following variable is designed for online processing
            DataReader<FeatType, LabelType>* reader;

        public:
            DataSet(size_t passes = 1, size_t buf_size = 1024)
            {
                this->head = NULL;
                this->wt_ptr = NULL;
                this->rd_ptr = NULL;
                this->reader = NULL;

                this->passNum = passes; 
                this->dataNum = 0;

                this->CreateBuffer(buf_size);
            }

        public:
            bool CreateBuffer(size_t buf_size = 1024)
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
            //bind a data reader to the dataset
            bool Load(DataReader<FeatType, LabelType> &reader)
            {
                this->reader = &reader;
                if (reader.OpenReading() == true)
                {
                    DataPoint<FeatType, LabelType> &data = this->GetDataWt();
                    for (size_t pass= 0; pass < this->passNum; pass++)
                    {
                        reader.Rewind();
                        while(true)
                        {
                            data.erase();
                            if (reader.GetNextData(data) == true)
                                data = this->GetDataWt();
                            else
                                break;
                        }
                    }
                    return true;
                }
                return false;
            }

            /////////////Data Access/////////////////////
        public:
            //get the data to read
            inline const DataPoint<FeatType, LabelType>& GetDataRd() const
            {
                const DataPoint<FeatType,LabelType>& data = this->rd_ptr->data;
                this->rd_ptr = this->rd_ptr->next;
                return data;
            }

            //get data to write
            inline DataPoint<FeatType, LabelType>& GetDataWt()
            {
                DataPoint<FeatType, LabelType> &data = this->wt_ptr->data;
                this->wt_ptr = this->wt_ptr->next;
                this->dataNum++;
                return data;
            }

            //the number of features
            inline size_t size() const {return this->dataNum; }
    };
}
