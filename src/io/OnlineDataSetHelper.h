/*************************************************************************
> File Name: DataSetHelper.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: Thu 24 Oct 2013 03:33:10 PM
> Descriptions: thread function definitions
************************************************************************/
#ifndef HEADER_DATASET_HELPER
#define HEADER_DATASET_HELPER


#include "binary_io.h"
#include "../utils/thread_primitive.h"
#include "DataChunk.h"

namespace BOC{
    //point type
#define PtType DataPoint<T1, T2>

	template <typename T1, typename T2> class OnlineDataSet;

	//load a chunk of data, return if file ended
	template <typename T1, typename T2>
	bool load_chunk(DataReader<T1, T2>* reader, FixSizeDataChunk<PtType >&chunk){
		bool not_file_end = true;
		chunk.erase();
		while (chunk.dataNum < chunk.chunk_size && not_file_end == true){
			DataPoint<T1, T2> &data = chunk.data[chunk.dataNum];
			not_file_end = reader->GetNextData(data);
			if (not_file_end == true)
				chunk.dataNum++;
			else
				break;
		}
		return not_file_end;
	}

	/**
	 * @Synopsis save_chunk save a chunk of data to disk
	 *
	 * @tparam T1 type of feature
	 * @tparam T2 type of label
	 * @Param writer writer hander
	 * @Param chunk data to be written
	 *
	 * @Returns  true if saved successfully
	 */
	template <typename T1, typename T2>
	bool save_chunk(binary_io<T1, T2> *writer, FixSizeDataChunk<PtType>&chunk){
		size_t w_num = 0;
		while (w_num < chunk.dataNum){
			if (writer->WriteData(chunk.data[w_num]) == true)
				w_num++;
			else
				return false;
		}
		return true;
	}

	/**
	 * @Synopsis get_cacher get a writer for caching file
	 *
	 * @tparam T1 type of feature
	 * @tparam T2 type of label
	 * @Param cache_filename specified cache-file name
	 *
	 * @Returns
	 */
	template <typename T1, typename T2>
	binary_io<T1, T2>* get_cacher(const std::string &cache_filename){
		string tmpFileName = cache_filename + ".writing";
		binary_io<T1, T2>* cacher = new binary_io<T1, T2>(tmpFileName);
		if (cacher->OpenWriting() == false){
			cerr << "Open cache file failed!" << endl;
			delete cacher;
			return NULL;
		}
		return cacher;
	}

	/**
	 * @Synopsis end_cache post processing for the cached file, delete the
	 * writer and rename the current file currently
	 *
	 * @tparam T1
	 * @tparam T2
	 * @Param cacher
	 * @Param cache_filename
	 *
	 * @Returns
	 */
	template <typename T1, typename T2>
	bool end_cache(binary_io<T1, T2>**cacher, const std::string& cache_filename){
		string tmpFileName = (*cacher)->get_filename();
		(*cacher)->Close();
		delete *cacher;
		*cacher = NULL;
		return rename_file(tmpFileName, cache_filename);
	}

	/**
	 * @Synopsis CacheLoad Load and cache  the dataset
	 *
	 * @tparam T1   type of feature
	 * @tparam T2   type of label
	 * @Param dataset
	 *
	 * @Returns true if load and cached successfully
	 */
	template <typename T1, typename T2>
	bool CacheLoad(OnlineDataSet<T1, T2> *dataset){
		DataReader<T1, T2>* reader = dataset->reader;
		reader->Rewind();
		if (reader->Good() == false) {
			cerr << "reader is incorrect!" << endl;
			return false;
		}

		binary_io<T1, T2>* writer = get_cacher<T1, T2>(dataset->cache_filename);
		if (writer == NULL)
			return false;

		//load data
		bool not_file_end = false;
		do {
			FixSizeDataChunk<PtType> &chunk = dataset->GetWriteChunk();
			not_file_end = load_chunk(reader, chunk);
			if (save_chunk(writer, chunk) == false){
				dataset->EndWriteChunk(chunk);
				break;
			}
			dataset->EndWriteChunk(chunk);
		} while (not_file_end == true);
		if (reader->Good() && writer->Good())
			return end_cache(&writer, dataset->cache_filename);
		else
			return false;
	}

	/* --------------------------------------------------------------------------*/
	/**
	 * @Synopsis thread_LoadData load dataset in parallel
	 *
	 * @tparam T1   type of feature
	 * @tparam T2   type of label
	 * @Param  param    pointer to the dataset class
	 *
	 * @Returns null
	 */
	/* ----------------------------------------------------------------------------*/
	template <typename T1, typename T2>
#if WIN32
	DWORD WINAPI thread_LoadData(LPVOID param)
#else
	void* thread_LoadData(void* param)
#endif
	{
		OnlineDataSet<T1, T2>* dataset = static_cast<OnlineDataSet<T1, T2>*>(param);
		DataReader<T1, T2>* reader = dataset->reader;

		int pass = 0;
		//if load dataset and cache the dataset
		if (dataset->is_cache == true){
			if (CacheLoad(dataset) == false){
				cerr << "caching data failed!" << endl;
				dataset->FinishParse();
				return NULL;
			}
			pass++;
			//multi-pass
			if (pass < dataset->pass_num){
				//setup the new cache-file reader
				dataset->delete_reader();
				dataset->self_reader = new binary_io<T1, T2>(dataset->cache_filename);
				dataset->reader = dataset->self_reader;
				dataset->is_cache = false;
				if (dataset->reader->OpenReading() == false){
					cerr << "load cache data failed!" << endl;
					dataset->FinishParse();
					return NULL;
				}
				reader = dataset->reader;
			}
		}

		//online algorithms will run multiple times
		for (; pass < dataset->pass_num; pass++) {
			reader->Rewind();
			if (reader->Good()) {
				bool not_file_end = false;
				do {
					FixSizeDataChunk<PtType> &chunk = dataset->GetWriteChunk();
					not_file_end = load_chunk(reader, chunk);
					dataset->EndWriteChunk(chunk);
				} while (not_file_end == true);
				if (reader->Good() == false) {
					cerr << "Load cached dataset failed!" << endl;
					break;
				}
			}
			else {
				cerr << "reader is incorrect!" << endl;
				break;
			}
		}
		dataset->FinishParse();
		return NULL;
	}
}
#endif
