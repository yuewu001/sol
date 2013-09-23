/*************************************************************************
  > File Name: DataPoint.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: 2013/8/18 星期日 20:13:31
  > Functions: Data Point Definition
 ************************************************************************/

#pragma once

#include "s_array.h"

#include <cstring>

namespace SOL
{
    static const int chunk_buf_size = 256; 
    /**
     *  Definitions of DataPoint: one lable, and DataPoints
     *
     * @tparam DataType
     */
    template <typename FeatType, typename LabelType> class DataPoint
    {
        public:
            //////////////Member Variables
            s_array<size_t> indexes;
            s_array<FeatType> features;
            s_array<float> weights;
            LabelType label;

            //for copy and release control
            int *count;

        public:
            DataPoint()
            {
                this->count = new int;
                *count = 1;
            }

            //copy constructor
            DataPoint(const DataPoint &point)
            {
                this->indexes = point.indexes;
                this->features = point.features;
                this->weights = point.weights;
                this->label = point.label;
                this->count = point.count;
                ++(*count);
            }

            ~DataPoint(){this->release();} 

            //assignment
            DataPoint<FeatType, LabelType>& operator= (const DataPoint<FeatType, LabelType> &data)
            {
                if (data.count == this->count)
                    return *this;
                this->release();

                this->indexes = data.indexes;
                this->features = data.features;
                this->weights = data.weights;
                this->label = data.label;
                this->count = data.count;
                ++(*count);
                return *this;
            }
            //set new index-value pair
            void AddNewFeat(const size_t &index, const FeatType &feat, const float weight = 0)
            {
                this->indexes.push_back(index);
                this->features.push_back(feat);
            }

            void erase()
            {
                this->indexes.erase();
                this->features.erase();
                this->weights.erase();
            }

        private:
            void release()
            {
                --(*count); 
                if (*count == 0)
                    delete count;
                this->count = NULL;
            }

    };
    template <typename FeatType, typename LabelType> struct DataChunk
    {
        DataPoint<FeatType, LabelType> data[chunk_buf_size];
        size_t dataNum;
        DataChunk *next;

        DataChunk():dataNum(0),next(NULL){}
        void erase()
        {
            for (int i = 0; i < dataNum; i++)
                data[i].erase();
            dataNum = 0;
        }
    };

}
