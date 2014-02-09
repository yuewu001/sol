/*************************************************************************
	> File Name: MPBuffer.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2/8/2014 7:38:57 PM
	> Functions: Buffer for multi-pass
 ************************************************************************/

#ifndef HEADER_MP_BUFFER
#define HEADER_MP_BUFFER

#include "DataPoint.h"

#include <ctime>

namespace SOL{

	enum MPBufferType{
		MPBufferType_None = 0, //no buffer
		MPBufferType_ALL = 1, //buffer all data
		MPBufferType_FALSE_PREDICT = 2, //buffer false predictions
		MPBufferType_RESERVIOR = 3, //reservior sampling
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
		template <typename FeatType, typename LabelType>
		struct MPBuffer_RESERVIOR :public MPBuffer<FeatType, LabelType>{
			size_t total_num; //total number of pushed instances
			MPBuffer_RESERVIOR(size_t bufSize = init_mp_buf_size) :MPBuffer(bufSize){
				//srand(time(NULL));
				this->total_num = 0;
			}

			void Push(DataPoint<FeatType, LabelType> &srcPt){
				this->total_num++;
				if (this->dataNum < this->chunk_size){ //less than the buffer
					srcPt.clone(this->data[this->insert_pos]);
					this->insert_pos++;
					this->dataNum++;
				}
				else{//sampling
					//sample from a Bernoulli distribution to determine whether we need to insert the current sample
					//if (this->bernolli(this->chunk_size,this->total_num)){
					{
						this->insert_pos = rand() % this->chunk_size;
						srcPt.clone(this->data[this->insert_pos]);
					}
				}
			}
			bool bernolli(size_t limit, size_t max_num) {
				static double max1 = RAND_MAX + 1.0;
				static double max2 = std::pow(2, 32);
				if (max_num < RAND_MAX)
					return rand() / max1 * max_num < limit;
				else
					return (((rand() & 0x00007FE0) >> 5) + ((rand() & 0x00007FF0) << 6) + ((rand() & 0x00007FF0) << 17)) 
					/ max2 * max_num < limit;
			}
		};
}

#endif

