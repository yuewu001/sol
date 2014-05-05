/*************************************************************************
	> File Name: OnlineDataSet.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 5/4/2014 10:32:53 PM
	> Functions: dataset which loads data in parallel with online algorithms
 ************************************************************************/
#ifndef HEADER_ONLINE_DATASET
#define HEADER_ONLINE_DATASET

#if WIN32
#include <windows.h>
#endif

#include "io_helper.h"
#include "OnlineDataSetHelper.h"

#include "DataSet.h"

using namespace std;

/**
 *  namespace: Sparse Online Learning
 */
namespace SOL {
    //data set, can work in both read-and-write mode and read-once mode
    template <typename FeatType, typename LabelType> class OnlineDataSet : public DataSet<FeatType, LabelType> {		
        protected:
            int bufSize; //buffer to load data
            int passNum; //number of passes

            //pointer to the first element, circlar linked list will be used
            FixSizeDataChunk<FeatType,LabelType> *head; 
            FixSizeDataChunk<FeatType,LabelType> *wt_ptr; //pointer to the write location
            FixSizeDataChunk<FeatType,LabelType> *rd_ptr; //pointer to the read location

            bool load_finished; //this is used for GetChunk to test if current loading has finished
            bool is_on_loading; //this is used for Rewind to test if rewind can be performed

            DataReader<FeatType,LabelType> *reader;
			bool is_reader_self_alloc;

            //thread-safety
            MUTEX data_lock;
            CV data_available;
            CV buffer_full;

			//double time1;
			//double time2;
			//double load_time;

        public:
            OnlineDataSet(int passes = 1, int buf_size = -1) {
                this->head = NULL;
                this->wt_ptr = NULL;
                this->rd_ptr = NULL;

                this->passNum = passes > 0 ? passes : 1; 
                this->dataNum = 0;
                this->curChunkNum = 0;

                this->load_finished = false;
                this->is_on_loading = false;
                this->reader = NULL;
                this->is_cache = false;
				this->is_reader_self_alloc = false;

                this->CreateBuffer(buf_size);

                //init thread-safety 
                initialize_mutex(&this->data_lock);
                initialize_condition_variable(&data_available);
                initialize_condition_variable(&buffer_full);

				//this->load_time = 0;
            }
            virtual ~OnlineDataSet() {
                delete_mutex(&data_lock);
				if (this->reader != NULL && this->is_reader_self_alloc == true)
                    delete this->reader;
				this->reader = NULL;
				this->ReleaseBuffer();
            }

        protected:
            bool CreateBuffer(int buf_size = 0) {
                this->ReleaseBuffer();
                this->bufSize = buf_size > 0 ? buf_size : init_buf_size;
                if (this->bufSize <= 0)
                    return true;

                this->head = new FixSizeDataChunk<FeatType,LabelType>(init_chunk_size);
                FixSizeDataChunk<FeatType,LabelType> *p = this->head;
                for (int i = 1; i < this->bufSize; i++) {
                    p->next = new FixSizeDataChunk<FeatType,LabelType>(init_chunk_size);
                    p = p->next;
                }
                p->next = this->head;
                this->wt_ptr = this->head;
                this->rd_ptr = this->head;

                return true;
            }

        protected:
            void ClearBuffer() {
                FixSizeDataChunk<FeatType,LabelType> *p = this->head;
                if (p == NULL)
                    return;
                p = p->next;
                while (p != this->head) {
                    p->erase();
                    p = p->next;
                }
                p->erase();
                this->dataNum = 0;
                this->curChunkNum = 0;
                this->wt_ptr = this->head;
                this->rd_ptr = this->head;
            }

            void ReleaseBuffer() {
                FixSizeDataChunk<FeatType,LabelType> *p = this->head;
                if (p == NULL)
                    return;
                FixSizeDataChunk<FeatType,LabelType> *q = p->next;
                while (q != this->head) {
                    p = q->next;
                    delete q;
                    q = p;
                }
                delete this->head;
                this->head = NULL;
                this->wt_ptr = NULL;
                this->rd_ptr = NULL;
                this->dataNum = 0;
            } 

        public:
			template <typename T1, typename T2> friend bool CacheLoad(OnlineDataSet<T1, T2> *dataset);
#if WIN32
            template <typename T1, typename T2> friend DWORD WINAPI thread_LoadData(LPVOID param);
#else
            template <typename T1, typename T2> friend void* thread_LoadData(void* param);
#endif
			//Load cached dataset
			bool Load(const string& cache_filename) {
				this->cache_fileName = cache_filename;
				if (this->reader != NULL && this->is_reader_self_alloc == true)
					delete this->reader;
				this->is_cache = false;
				this->reader = new libsvm_binary_<FeatType, LabelType>(this->cache_fileName);
				this->is_reader_self_alloc = true;

				if (this->reader != NULL){
					if (this->reader->OpenReading() == false){
						delete this->reader;
						this->reader = NULL;
						return false;
					}
				}
				return true;
			}

			//bind a data reader to the dataset
			bool Load(DataReader<FeatType, LabelType> *ext_reader) {
				if (this->reader != NULL && this->is_reader_self_alloc == true)
						delete this->reader;
					this->reader = ext_reader;
					this->is_cache = false;
					this->is_reader_self_alloc = false;

					if (this->reader != NULL){
						if (this->reader->OpenReading() == false){
							return false;
						}
					}
					return true;
			}

			//bind a data reader to the dataset
			bool Load(DataReader<FeatType, LabelType> *ext_reader,
				const string& cache_filename) {
					if (SOL_ACCESS(cache_filename.c_str()) == 0){ //already cached
						return this->Load(cache_filename);
					}
					else if(ext_reader != NULL){
						if (this->Load(ext_reader) == false){
							return false;
						}
						if (cache_filename.length() == 0 && this->passNum > 1){ 
							this->cache_fileName = "cache_file";
							this->is_cache = true;
						}
						else if (cache_filename.length() > 0){
							this->cache_fileName = cache_filename;
							this->is_cache = true;
						}
					}
					else
						return false;
					
					return true;
			}

			//bind a data reader to the dataset
			bool Load(const string& filename,  const string& cache_filename, const string &dt_type = "libsvm") {
				if (SOL_ACCESS(cache_filename.c_str()) == 0){ //already cached
					return this->Load(cache_filename);
				}
				else {
					this->fileName = filename;
					DataReader<FeatType, LabelType>* new_reader = getReader<FeatType, LabelType>(filename, dt_type);
					bool ret = this->Load(new_reader, cache_filename);
					this->is_reader_self_alloc = true;
					return ret;
				}
			}

			/////////////Data Access/////////////////////
		public:

			//get the next write chunk
			inline FixSizeDataChunk<FeatType, LabelType> &GetWriteChunk(){
				mutex_lock(&this->data_lock); 
				if (this->wt_ptr->is_inuse == false){
					this->wt_ptr->is_inuse = true;
					FixSizeDataChunk<FeatType, LabelType>* p = this->wt_ptr;
					mutex_unlock(&this->data_lock);
					//this->time1 = get_current_time();
					return *p;
				}
				else{
					condition_variable_wait(&this->buffer_full,&this->data_lock);
					mutex_unlock(&this->data_lock);
					return this->GetWriteChunk();
				}
			}

			inline void EndWriteChunk(){
				mutex_lock(&this->data_lock);
				this->wt_ptr->is_parsed = true;
				this->dataNum += this->wt_ptr->dataNum;
				//if (this->wt_ptr->dataNum == 0){
				//	cout<<"chunk size is zero!"<<endl;
				//}
				this->wt_ptr = this->wt_ptr->next;
				condition_variable_signal_all(&this->data_available);
				//this->time2 = get_current_time();
				//this->load_time += time2 - time1;
				mutex_unlock(&this->data_lock);
			}

			inline void FinishParse(){
				//notice that the all the data has been loaded
				mutex_lock(&this->data_lock);
				this->load_finished = true;
				this->is_on_loading = false;
				condition_variable_signal_all(&this->data_available);
				//cout<<"loading time: "<<this->load_time<<" s"<<endl;
				mutex_unlock(&this->data_lock);
			}

			//get the data to read
			virtual DataChunk<FeatType, LabelType>& GetChunk(bool is_test = false) {
				mutex_lock(&this->data_lock);
				//check if there is available data
				if (this->rd_ptr->is_parsed == true){
					this->rd_ptr->is_parsed = false;
					mutex_unlock(&this->data_lock);
					return *(this->rd_ptr);
				}
				else{ //no available data 
					if (this->load_finished == true){
						this->rd_ptr->is_parsed = false;
						this->rd_ptr->erase();
						mutex_unlock(&this->data_lock);
						return *(this->rd_ptr); //return an invalid data
					}
					else{ //suspend the current thread
						condition_variable_wait(&this->data_available,&this->data_lock);
						mutex_unlock(&this->data_lock);
						return this->GetChunk();
					}
				}
			}

			virtual void FinishRead() {
				mutex_lock(&this->data_lock);
				this->rd_ptr->is_inuse = false;
				//notice that the last data have been processed
				this->rd_ptr = this->rd_ptr->next;
				condition_variable_signal_all(&this->buffer_full);
				mutex_unlock(&this->data_lock);
			}

			bool Rewind() {
				mutex_lock(&this->data_lock);
				if (this->is_on_loading == true) {
					cout<<"data is on loading"<<endl;
					mutex_unlock(&this->data_lock);
					return false;
				}
				reader->Rewind();
				this->ClearBuffer();
				this->load_finished = false;
				this->is_on_loading = true;
				mutex_unlock(&this->data_lock);

#if WIN32
				HANDLE thread = ::CreateThread(NULL, 0, static_cast<LPTHREAD_START_ROUTINE>(thread_LoadData<FeatType,LabelType>), this, NULL, NULL);
#else
				pthread_t thread;
				pthread_create(&thread,NULL,thread_LoadData<FeatType,LabelType>,this);
#endif
				return true;
			}
	};
}


#endif

