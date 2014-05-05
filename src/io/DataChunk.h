/*************************************************************************
	> File Name: DataChunk.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 5/5/2014 11:25:18 AM
	> Functions: chunk of data
 ************************************************************************/

#ifndef HEADER_DATA_CHUNK
#define HEADER_DATA_CHUNK

#include "DataPoint.h"

#include <vector>
using std::vector;

namespace BOC {
	template <typename FeatType, typename LabelType> 
    struct DataChunk{
		vector<DataPoint<FeatType, LabelType> > data;
		size_t dataNum;

        DataChunk():dataNum(0){}
        };

	template <typename FeatType, typename LabelType> 
	struct FixSizeDataChunk: public DataChunk<FeatType, LabelType> {
		size_t chunk_size;
		FixSizeDataChunk *next;
		bool is_inuse;
		bool is_parsed;

		FixSizeDataChunk(size_t chunkSize) : chunk_size(chunkSize),
			next(NULL), is_inuse(false), is_parsed(false){
			if (this->chunk_size == 0){
				std::cerr << "error occured at file: " << __FILE__ << ": line" << __LINE__ <<
					"\nERROR: chunk size for multi-pass should be a positive!" << std::endl;
				exit(2);
			}
			try{
				this->data.resize(this->chunk_size);// = new DataPoint<FeatType, LabelType>[this->chunk_size];
			}
			catch (std::bad_alloc &ex){
				std::cerr << ex.what();
				std::cerr << " allocate of " << this->chunk_size
					<< " failed in constructing DataChunk. out of memory? in file "
					<< __FILE__ << " line " << __LINE__ << std::endl;
				exit(1);
			}
		}

		virtual ~FixSizeDataChunk(){
		}

        /**
         * @Synopsis erase Erase all the data in chunk
         */
		void erase() {
			for (size_t i = 0; i < this->chunk_size; i++)
				this->data[i].erase();
			this->dataNum = 0;
		}
	};
}

#endif

