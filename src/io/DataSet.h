/*************************************************************************
  > File Name: DataSet.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: 2013/8/18 Sunday 15:38:09
  > Functions: Class to interact with datasets
 ************************************************************************/

#ifndef HEADER_DATASET
#define HEADER_DATASET

#include "DataPoint.h"
#include "DataChunk.h"
#include "DataReader.h"

using namespace std;

/**
 *  namespace: Sparse Online Learning
 */
namespace SOL {
    //data set, can work in both read-and-write mode and read-once mode
    template <typename FeatType, typename LabelType> class DataSet {		
        protected:
            string fileName;
            string cache_fileName;
            bool is_cache;

            size_t dataNum; //total data number
            size_t curChunkNum;  //data number in buffer

        public:
            virtual ~DataSet() {
                this->dataNum = 0;
                this->curChunkNum = 0;
                this->is_cache = false;
            }

        public:
            //Load cached dataset
            virtual bool Load(const string& cache_filename)  = 0;

            //bind a data reader to the dataset
            virtual bool Load(DataReader<FeatType, LabelType> *ext_reader)  = 0;

            //bind a data reader to the dataset
            virtual bool Load(DataReader<FeatType, LabelType> *ext_reader, const string& cache_filename) = 0;

            //bind a data reader to the dataset
            virtual bool Load(const string& filename,  const string& cache_filename, const string &dt_type = "libsvm")  = 0;

            /////////////Data Access/////////////////////
        public:
            //get the data to read
            virtual DataChunk<FeatType, LabelType>& GetChunk(bool is_test = false) = 0;

            virtual void FinishRead() = 0;

            //the number of features
            inline size_t size() const {return this->dataNum; }

            //rewind the dataset
            virtual bool Rewind() = 0;
    };
}

#endif
