/*************************************************************************
	> File Name: MPBuffer.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2/8/2014 7:38:57 PM
	> Functions: Buffer for multi-pass
 ************************************************************************/

#ifndef HEADER_MP_BUFFER
#define HEADER_MP_BUFFER

#include "DataPoint.h"

namespace SOL{

	enum MPBufferType{
		MPBufferType_None = 0, //no buffer
		MPBufferType_ALL = 1, //buffer all data
		MPBufferType_FALSE_PREDICT = 2, //buffer false predictions
	};

    template <typename FeatType, typename LabelType> 
        struct MPBuffer:public DataChunk<FeatType,LabelType>{
            size_t insert_pos;

            MPBuffer(size_t bufSize = init_mp_buf_size):DataChunk(bufSize),insert_pos(0){
				this->is_inherited = true;
            }

			virtual void Push(DataPoint<FeatType, LabelType> &srcPt) = 0;
		};

		template <typename FeatType, typename LabelType>
		struct MPBuffer_ALL :public MPBuffer<FeatType, LabelType>{
			MPBuffer_ALL(size_t bufSize = init_mp_buf_size) :MPBuffer(bufSize){
			}

			void Push(DataPoint<FeatType, LabelType> &srcPt){
				if (this->insert_pos == this->chunk_size){
					this->insert_pos = 0;
				}
				srcPt.clone(this->data[this->insert_pos]);
				this->insert_pos++;
				if (this->dataNum < this->chunk_size)
					this->dataNum++;
			}
		};
		template <typename FeatType, typename LabelType>
		struct MPBuffer_FALSE_PREDICT :public MPBuffer<FeatType, LabelType>{
			MPBuffer_FALSE_PREDICT(size_t bufSize = init_mp_buf_size) :MPBuffer(bufSize){
			}

			void Push(DataPoint<FeatType, LabelType> &srcPt){
				if (this->insert_pos == this->chunk_size){
					this->insert_pos = 0;
				}
				if (srcPt.loss > 0) {
					srcPt.clone(this->data[this->insert_pos]);
					this->insert_pos++;
					if (this->dataNum < this->chunk_size)
						this->dataNum++;
				}
			}
		};
}

#endif

