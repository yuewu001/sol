/*************************************************************************
> File Name: DataPoint.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/18 Sunday 20:13:31
> Functions: Data Point Definition
************************************************************************/

#ifndef HEADER_DATA_POINT
#define HEADER_DATA_POINT

#include "s_array.h"
#include "../utils/init_param.h"

#include <cstring>

namespace SOL {
	/**
	*  Definitions of DataPoint: one lable, and DataPoints
	*
	* @tparam DataType
	*/
	template <typename FeatType, typename LabelType> 
	class DataPoint {
	public:
		//////////////Member Variables
		s_array<IndexType> indexes;
		s_array<FeatType> features;
		LabelType label;
		FeatType sum_sq; //sum of square
		float margin;

		//for copy and release control
		int *count;

		IndexType max_index; //max index, also the dimension
	public:
		DataPoint() {
			this->count = new int;
			*count = 1;
			this->max_index = 0;
			this->label = 0;
			this->sum_sq = 0;
			this->margin= 0;
		}

		//copy constructor
		DataPoint(const DataPoint &point) {
			this->indexes = point.indexes;
			this->features = point.features;
			this->label = point.label;
			this->count = point.count;
			this->max_index = point.max_index;
			this->sum_sq = point.sum_sq;
			this->margin= point.margin;
			++(*count);
		}

		~DataPoint(){
			this->release();
		} 

		//assignment
		DataPoint<FeatType, LabelType>& operator= 
			(const DataPoint<FeatType, LabelType> &data) {
				if (data.count == this->count)
					return *this;
				this->release();

				this->indexes = data.indexes;
				this->features = data.features;
				this->label = data.label;
				this->max_index = data.max_index;
				this->sum_sq = data.sum_sq;
				this->margin = data.margin;
				this->count = data.count;
				++(*count);
				return *this;
		}
		//set new index-value pair
		void AddNewFeat(const IndexType &index, 
			const FeatType &feat) {
				this->indexes.push_back(index);
				this->features.push_back(feat);
				if(this->max_index < index){
					this->max_index = index;
				}
				this->sum_sq += feat * feat;
		}

		void erase() {
			this->indexes.erase();
			this->features.erase();
			this->max_index = 0;
			this->sum_sq = 0;
			this->margin = 0;
		}
		void clone(DataPoint<FeatType, LabelType> &dstPt) const{
			dstPt.label = this->label;
			dstPt.max_index = this->max_index;
			dstPt.sum_sq = this->sum_sq;
			dstPt.margin = this->margin;
			dstPt.indexes.resize(this->indexes.size());
			memcpy(dstPt.indexes.begin, this->indexes.begin, this->indexes.size() * sizeof(IndexType));
			dstPt.features.resize(this->features.size());
			memcpy(dstPt.features.begin, this->features.begin, this->features.size() * sizeof(FeatType));
		}

		DataPoint<FeatType, LabelType> clone() const{
			DataPoint<FeatType, LabelType> newPt; 
			newPt.label = this->label;
			newPt.max_index = this->max_index;
			newPt.sum_sq = this->sum_sq;
			newPt.margin = this->margin;
			newPt.indexes.resize(this->indexes.size());
			memcpy(newPt.indexes.begin,this->indexes.begin, this->indexes.size() * sizeof(IndexType) );
			newPt.features.resize(this->features.size());
			memcpy(newPt.features.begin, this->features.begin, this->features.size() * sizeof(FeatType));
			return newPt;				
		}

		IndexType dim() const {return this->max_index;}

		bool is_sorted() const {
			bool sorted = true;
			if (this->indexes.size() <= 1)
				return true;
			for (IndexType *iter = this->indexes.begin + 1, *iter0 = this->indexes.begin; iter != this->indexes.end; iter0++,iter++){
				if (*iter0 > *iter){
					sorted = false;
					break;
				}
			}
			return sorted;
		}
		void Sort() {
			if (this->is_sorted() == true)
				return;
			QuickSort(this->indexes.begin,this->features.begin, 0,this->indexes.size() -1 );
		}

	private:
		void release() {
			--(*count); 
			if (*count == 0)
				delete count;
			this->count = NULL;
		}

	};
	template <typename FeatType, typename LabelType> 
	struct DataChunk{
		DataPoint<FeatType, LabelType> *data;
		size_t dataNum;
		size_t chunk_size;
		DataChunk *next;
		bool is_inuse;
		bool is_parsed;
		bool is_inherited; //judge if the class is inherited from DataChunk

		DataChunk(size_t chunkSize = init_chunk_size) :dataNum(0), chunk_size(chunkSize),
			next(NULL), is_inuse(false), is_parsed(false), is_inherited(false){
			if (this->chunk_size == 0){
				std::cerr << "error occured at file: " << __FILE__ << ": line" << __LINE__ <<
					"\nERROR: chunk size for multi-pass should be a positive!" << std::endl;
				exit(2);
			}
			try{
				this->data = new DataPoint<FeatType, LabelType>[this->chunk_size];
			}
			catch (std::bad_alloc &ex){
				std::cerr << ex.what();
				std::cerr << " allocate of " << this->chunk_size
					<< " failed in constructing DataChunk. out of memory? in file "
					<< __FILE__ << " line " << __LINE__ << std::endl;
				exit(1);
			}
		}
		virtual ~DataChunk(){
			if (this->data != NULL)
				delete[]this->data;
		}
		void erase() {
			for (size_t i = 0; i < this->chunk_size; i++)
				data[i].erase();
			dataNum = 0;
		}
	};

}
#endif
