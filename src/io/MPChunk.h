/*************************************************************************
	> File Name: MPChunk.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2/8/2014 7:38:57 PM
	> Functions: Buffer for multi-pass
 ************************************************************************/

#ifndef HEADER_MP_BUFFER
#define HEADER_MP_BUFFER

#include "DataPoint.h"
#include "DataChunk.h"

#include "../utils/MaxHeap.h"
#include "../utils/init_param.h"

#include <time.h>
#include <limits>
#include <cmath>

namespace BOC{

	enum MPChunkType{
		MPChunkType_None = 0, //no buffer
		MPChunkType_ALL = 1, //buffer all data
		MPType_MARGIN = 2, //buffer false predictions
		MPType_RESERVIOR = 3, //reservior sampling
	};

    template <typename PointType> 
        struct MPChunk : public FixSizeDataChunk<PointType>{
            size_t insert_pos;

            MPChunk(size_t bufSize = init_mp_buf_size) : FixSizeDataChunk<PointType>(bufSize),insert_pos(0){
            }

			virtual void Push(PointType &srcPt) = 0;
		};

		template <typename PointType>
		struct MPChunk_ALL :public MPChunk<PointType>{
			MPChunk_ALL(size_t bufSize = init_mp_buf_size) : MPChunk<PointType>(bufSize){
			}

			void Push(PointType &srcPt){
				if (this->insert_pos == this->chunk_size){
					this->insert_pos = 0;
				}
				srcPt.clone(this->data[this->insert_pos]);
				this->insert_pos++;
				if (this->dataNum < this->chunk_size)
					this->dataNum++;
			}
		};

		template <typename PointType>
		struct MPChunk_LARGE_MARGIN : public MPChunk<PointType>{
		private:
			MaxHeap<float, size_t> marginHeap;
			s_array<float> marginVec;
			s_array<int> bufIdVec;
			static float max_float;
		public:
			MPChunk_LARGE_MARGIN(size_t bufSize = init_mp_buf_size) : MPChunk<PointType>(bufSize){
				this->marginVec.reserve(bufSize);
				this->marginVec.resize(bufSize);
				for (s_array<float>::iterator iter = this->marginVec.begin;
					iter != this->marginVec.end; iter++)
					*iter = max_float;
				//this->marginHeap.Init(1,1, this->marginVec.begin);
			}

			void Push(PointType &srcPt){
				static size_t ret_id;
				if (this->dataNum < this->chunk_size){
					this->dataNum++;
					this->marginHeap.UpdateDataNum(this->dataNum, this->marginVec.begin);
					this->marginHeap.ResizeHeap(this->dataNum);
				}

				if (srcPt.margin < this->marginHeap.GetHeapLimit()){
					this->insert_pos = this->marginHeap.get_id(0);
					srcPt.clone(this->data[this->insert_pos]);
					this->marginVec[this->insert_pos] = srcPt.margin;
					this->marginHeap.UpdateHeap(this->insert_pos,ret_id);
				}
			}
		};
		template <typename PointType>
		float MPChunk_LARGE_MARGIN<PointType>::max_float =
			(std::numeric_limits<float>::max)();

		template <typename PointType>
		struct MPChunk_RESERVIOR :public MPChunk<PointType>{
			size_t total_num; //total number of pushed instances
			MPChunk_RESERVIOR(size_t bufSize = init_mp_buf_size) : MPChunk<PointType>(bufSize){
				//srand(time(NULL));
				this->total_num = 0;
			}

			void Push(PointType &srcPt){
				this->total_num++;
				if (this->dataNum < this->chunk_size){ //less than the buffer
					srcPt.clone(this->data[this->insert_pos]);
					this->insert_pos++;
					this->dataNum++;
				}
				else{//sampling
					//sample from a Bernoulli distribution to determine whether we need to insert the current sample
					this->insert_pos = this->bernolli(this->total_num);
					if (this->insert_pos < this->chunk_size){
						srcPt.clone(this->data[this->insert_pos]);
					}
				}
			}
			size_t bernolli(size_t max_num) {
				static double max1 = RAND_MAX + 1.0;
				static double max2 = std::pow(2, 32);
				if (max_num < RAND_MAX)
					return size_t(rand() / max1 * max_num);
				else
					return size_t((((rand() & 0x00007FE0) >> 5) + ((rand() & 0x00007FF0) << 6) + ((rand() & 0x00007FF0) << 17)) 
					/ max2 * max_num);
			}
		};
}

#endif

