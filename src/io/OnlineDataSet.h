/*************************************************************************
	> File Name: OnlineDataSet.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 5/4/2014 10:32:53 PM
	> Functions: dataset which loads data in parallel with online algorithms
	************************************************************************/
#ifndef HEADER_ONLINE_DATASET
#define HEADER_ONLINE_DATASET


#include "OnlineBuffer.h"
#include "OnlineDataSetHelper.h"

#include "DataSet.h"

#include <sstream>

/**
 *  namespace: Sparse Online Learning
 */
namespace BOC {
	//data set, can work in both read-and-write mode and read-once mode
	template <typename FeatType, typename LabelType>
	class OnlineDataSet : public DataSet<FeatType, LabelType> {
		//for dynamic binding
		DECLARE_CLASS

	protected:
		typedef FixSizeDataChunk<FeatType, LabelType> ChunkType;

		int pass_num; //number of passes
		OnlineBuffer<ChunkType> online_buf;

#if WIN32
		HANDLE thread;
#else
		pthread_t thread;
#endif
		/**
		 * @Synopsis Constructors
		 */
	public:
		OnlineDataSet(int passes, int buf_size, int chunk_size) :
			online_buf(buf_size, chunk_size), DataSet<FeatType, LabelType>() {
				if (passes < 1) {
					std::ostringstream oss;
					oss << "number of passes should be no less than 1, while " << passes << " is specified!";
					throw std::runtime_error(oss.str());
				}
				this->pass_num = passes;
			}

		virtual ~OnlineDataSet() {
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
			return DataSet<FeatType, LabelType>::Load(fileName, cache_fileName, dt_format);
		}

		/**
		 * @Synopsis Load load data from an extern reader or the cached file,
		 * cache the reader if cache file not exists or multipass is specified
		 *
		 * @Param ext_reader
		 * @Param cache_filename
		 *
		 * @Returns
		 */
		virtual bool Load(DataReader<FeatType, LabelType> *ext_reader, const string& cache_fileName) {
			//already cached
			if (SOL_ACCESS(cache_fileName.c_str()) == 0) {
				this->delete_reader();
				this->cache_filename = cache_fileName;
				this->self_reader = (DataReader<FeatType, LabelType>*)
					Registry::CreateObject("binary", &this->cache_filename);

				return this->Load(this->self_reader);
			}
			//not cached, but ext_reader is ok
			else if (ext_reader != NULL) {
				if (cache_fileName.length() > 0){
					this->cache_filename = cache_fileName;
					return this->Load(ext_reader, true);
				}
				else if (this->pass_num > 1){
					this->cache_filename = "cache_file";
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
			return DataSet<FeatType, LabelType>::Load(ext_reader, Is_cache);
		}

		/**
		 * @Synopsis Data Access
		 */
	public:
		/**
		 * @Synopsis GetWriteChunk get the next write chunk
		 *
		 * @Returns reference to the chunk
		 */
		inline ChunkType& GetWriteChunk(){
			return this->online_buf.GetWriteChunk();
		}

		/**
		 * @Synopsis EndWriteChunk Finish writing a chunk
		 */
		inline void EndWriteChunk(const ChunkType& chunk){
			this->data_num += chunk.dataNum;
			this->online_buf.EndWriteChunk();
		}

		/**
		 * @Synopsis FinishParse Finish loading the data
		 */
		inline void FinishParse(){
			return this->online_buf.FinishParse();
		}

		/**
		 * @Synopsis GetChunk read a chunk from the buffer
		 *
		 * @Returns reference to a chunk of data
		 */
		virtual DataChunk<FeatType, LabelType>& GetChunk() {
			return this->online_buf.GetChunk();
		}

		/**
		 * @Synopsis FinishRead finished processing the read chunk
		 */
		inline virtual void FinishRead() {
			return this->online_buf.FinishRead();
		}

		template <typename T1, typename T2> friend bool CacheLoad(OnlineDataSet<T1, T2> *dataset);
#if WIN32
		template <typename T1, typename T2> friend DWORD WINAPI thread_LoadData(LPVOID param);
#else
		template <typename T1, typename T2> friend void* thread_LoadData(void* param);
#endif

		/**
		 * @Synopsis Rewind Reset the reader to the beginning
		 */
		virtual void Rewind() {
			if (this->online_buf.BeginWriteChunk() == true){
				this->reader->Rewind();
				this->threadLoad();
			}
		}

	protected:
		void threadLoad(){
#if WIN32
			create_thread(thread, static_cast<LPTHREAD_START_ROUTINE>(thread_LoadData<FeatType, LabelType>), this);
			//HANDLE thread = ::CreateThread(NULL, 0, static_cast<LPTHREAD_START_ROUTINE>(thread_LoadData<FeatType, LabelType>), this, NULL, NULL);
#else
			create_thread(thread, thread_LoadData<FeatType, LabelType>, this);
			//pthread_create(&thread, NULL, thread_LoadData<FeatType, LabelType>, this);
#endif
		}
	};

	//for dynamic binding
	template <typename FeatType, typename LabelType> 
	ClassInfo OnlineDataSet<FeatType, LabelType>::classInfo("dt_online", 
		"parallel loading dataset", OnlineDataSet<FeatType, LabelType>::CreateObject); 

	template <typename FeatType, typename LabelType>
	void*  OnlineDataSet<FeatType, LabelType>::CreateObject(void *param1, void* param2, void* param3) {
		return new OnlineDataSet<FeatType, LabelType>(int(param1), int(param2), int(param3));
	}
}


#endif

