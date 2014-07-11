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

#include "../utils/reflector.h"

#include <string>

/**
 *  namespace: Sparse Online Learning
 */
namespace BOC {

	//data set, can work in both read-and-write mode and read-once mode
	template <typename FeatType, typename LabelType>
	class DataSet : public Registry {
	protected:
		std::string filename;
		std::string cache_filename;
		bool is_cache;
		//whether normalize data
		bool is_norm;

		size_t data_num; //total data number

		DataReader<FeatType, LabelType> *reader;
		DataReader<FeatType, LabelType> *self_reader;
		typedef DataPoint<FeatType, LabelType> PointType;

	public:
		DataSet()
			: data_num(0), is_cache(false), is_norm(NULL), reader(NULL), self_reader(NULL)
		{ }

		virtual ~DataSet(){
			this->delete_reader();
		}

	protected:
		void delete_reader() {
			if (this->self_reader != NULL){
				delete this->self_reader;
				this->self_reader = NULL;
			}
		}

	public:
		/**
		 * @Synopsis Load load data from an text data file or the cached file,
		 * cache the reader if cache file not exists or multipass is specified
		 *
		 * @Param filename
		 * @Param cache_filename
		 *
		 * @Returns true if succeed
		 */
		virtual bool Load(const std::string& fileName, const std::string& cache_fileName, const std::string &dt_format) {
			//load from file
			if (SOL_ACCESS(fileName.c_str()) == 0) {
				this->delete_reader();
				this->filename = fileName;
				this->self_reader = (DataReader<FeatType, LabelType>*)
					Registry::CreateObject(dt_format, &this->filename);

				return this->Load(this->self_reader, cache_fileName);
			}
			//not exist 
			else {
				return this->Load(NULL, cache_fileName);
			}
		}

		/**
		 * @Synopsis Load load data from an extern reader or the cached file,
		 * cache the reader if cache file not exists or multipass is specified
		 *
		 * @Param ext_reader
		 * @Param cache_filename
		 *
		 * @Returns true if succeed
		 */
		virtual bool Load(DataReader<FeatType, LabelType> *ext_reader, const std::string& cache_filename) {
			//already cached
			if (SOL_ACCESS(cache_filename.c_str()) == 0) {
				this->delete_reader();
				this->cache_filename = cache_filename;
				this->self_reader = (DataReader<FeatType, LabelType>*)
					Registry::CreateObject("binary", &this->cache_filename);

				return this->Load(this->self_reader);
			}
			//not cached, but ext_reader is ok
			else if (ext_reader != NULL) {
				if (this->cache_filename.length() > 0){
					this->cache_filename = cache_filename;
					return this->Load(ext_reader, true);
				}
				else{
					this->cache_filename.clear();
					return this->Load(ext_reader, false);
				}
			}
			//not cached, ext_reader is null
			else{
				return false;
			}
		}

		/**
		 * @Synopsis Load Load the data fro ext_reader and no cache
		 *
		 * @Param ext_reader reader to load data from
		 *
		 * @Returns true if succeed
		 */
		virtual bool Load(DataReader<FeatType, LabelType> *ext_reader, bool Is_cache = false) {
			this->reader = ext_reader;
			this->is_cache = Is_cache;

			if (this->reader != NULL){
				if (this->reader->OpenReading() == false){
					return false;
				}
			}
			return true;
		}

		/**
		 * @Synopsis Data Access
		 */
	public:
		/**
		 * @Synopsis GetChunk read a chunk from the buffer
		 *
		 * @Returns reference to a chunk of data
		 */
		virtual DataChunk<PointType>& GetChunk() = 0;

		/**
		 * @Synopsis FinishRead finished processing the read chunk
		 */
		virtual void FinishRead() = 0;

		/**
		 * @Synopsis size number of features
		 *
		 * @Returns number of features
		 */
		inline size_t size() const { return this->data_num; }

		/**
		 * @Synopsis Rewind Reset the reader to the beginning
		 */
		virtual void Rewind() = 0;
	};
}

#endif
