/*************************************************************************
	> File Name: OnlineBuffer.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 5/6/2014 9:54:57 AM
	> Functions: buffer for online parallel processing of data
 ************************************************************************/
#ifndef HEADER_ONLINE_BUFFER
#define HEADER_ONLINE_BUFFER

#include "../utils/thread_primitive.h"

#include <stdexcept>

/**
 *  namespace: Sparse Online Learning
 */
namespace BOC {
    //data set, can work in both read-and-write mode and read-once mode
    template <typename ElemType> class OnlineBuffer {		
        protected:
            int buf_size; //buffer to load data
            int chunk_size; //number of samples in each chunk

            bool is_on_loading; //denote if the writer is on loading

            //pointer to the first element, circlar linked list will be used
            ElemType *head; 
            ElemType *wt_ptr; //pointer to the write location
            ElemType *rd_ptr; //pointer to the read location

            //thread-safety
            MUTEX data_lock;
            CV data_available;
            CV buffer_full;

            /**
             * @Synopsis Constructors
             */
        public:
            OnlineBuffer(int buf_size, int chunk_size) {
                this->head = NULL;
                this->wt_ptr = NULL;
                this->rd_ptr = NULL;

                this->is_on_loading = false;

                //init for thread-safety 
                initialize_mutex(&this->data_lock);
                initialize_condition_variable(&data_available);
                initialize_condition_variable(&buffer_full);

                if (this->CreateBuffer(buf_size, chunk_size) == false){
                    throw runtime_error("create buffer when initializing online buffer failed!");
                }
            }

            virtual ~OnlineBuffer() {
				this->ReleaseBuffer();
                delete_mutex(&data_lock);
            }

        protected:
            /**
             * @Synopsis CreateBuffer Create the specified size of buffer
             *
             * @Param buf_size  number of chunks in the buffer
             * @Param chunk_size    number of samples in each chunk
             *
             * @Returns true if succeed
             */
            bool CreateBuffer(int buf_size, int chunk_size) {
				mutex_lock(&this->data_lock); 
                if (this->ReleaseBuffer() == false){
                    mutex_unlock(&this->data_lock);
                    return false;
                }

                if (buf_size  <= 0 || chunk_size <= 0){
                    mutex_unlock(&this->data_lock);
                    return false;
                }

                this->buf_size = buf_size; 
                this->chunk_size = chunk_size;
                this->is_on_loading = true;

                this->head = new ElemType(chunk_size);
                ElemType *p = this->head;
                for (int i = 1; i < this->buf_size; i++) {
                    p->next = new ElemType(chunk_size);
                    p = p->next;
                }
                p->next = this->head;
                this->wt_ptr = this->head;
                this->rd_ptr = this->head;

				mutex_unlock(&this->data_lock);
                return true;
            }

            /**
             * @Synopsis ReleaseBuffer Release resources in the buffer
             *
             * @Returns true if succeed
             */
            bool ReleaseBuffer() {
				mutex_lock(&this->data_lock); 
                if (this->is_on_loading == true){
                    fprintf(stderr,"buffer is in use, cannot be released!");
                    mutex_unlock(&this->data_lock);
                    return false;
                }

                ElemType *p = this->head;
                if (p == NULL){
                    mutex_unlock(&this->data_lock);
                    return true;
                }

                ElemType *q = p->next;
                while (q != this->head) {
                    p = q->next;
                    delete q;
                    q = p;
                }
                delete this->head;
                this->head = NULL;
                this->wt_ptr = NULL;
                this->rd_ptr = NULL;

                this->buf_size = 0;
                this->chunk_size = 0;

                mutex_unlock(&this->data_lock);
                return true;
            } 

            /**
             * @Synopsis Data Access
             */
		public:
            /**
             * @Synopsis BeginWriteChunk Clear the content in each chunk
             *
             * @Returns true if succeed
             */
            bool BeginWriteChunk() {
				mutex_lock(&this->data_lock);
                if (this->is_on_loading == true){
                    //fprintf(stderr,"buffer is in use, cannot be released!");
                    mutex_unlock(&this->data_lock);
                    return false;
                }

                ElemType *p = this->head;
                if (p != NULL){
                    p = p->next;
                    while (p != this->head) {
                        p->erase();
                        p = p->next;
                    }
                    p->erase();
                }

                this->wt_ptr = this->head;
                this->rd_ptr = this->head;
				this->is_on_loading = true;
				mutex_unlock(&this->data_lock);

                return true;
            }

            /**
             * @Synopsis GetWriteChunk get the next write chunk
             *
             * @Returns reference to the chunk
             */
			inline ElemType& GetWriteChunk(){
				mutex_lock(&this->data_lock); 
				if (this->wt_ptr->is_inuse == false){
					this->wt_ptr->is_inuse = true;
					ElemType* p = this->wt_ptr;
					mutex_unlock(&this->data_lock);
					return *p;
				}
				else{
					condition_variable_wait(&this->buffer_full,&this->data_lock);
					mutex_unlock(&this->data_lock);
					return this->GetWriteChunk();
				}
			}

            /**
             * @Synopsis EndWriteChunk Finish writing a chunk
             */
			inline void EndWriteChunk(){
				mutex_lock(&this->data_lock);
				this->wt_ptr->is_parsed = true;
				//if (this->wt_ptr->dataNum == 0){
				//	cout<<"chunk size is zero!"<<endl;
				//}
				this->wt_ptr = this->wt_ptr->next;
				condition_variable_signal_all(&this->data_available);
				//this->time2 = get_current_time();
				//this->load_time += time2 - time1;
				mutex_unlock(&this->data_lock);
			}

            /**
             * @Synopsis FinishParse Finish loading the data
             */
			inline void FinishParse(){
				//notice that the all the data has been loaded
				mutex_lock(&this->data_lock);
				this->is_on_loading = false;
				condition_variable_signal_all(&this->data_available);
				//cout<<"loading time: "<<this->load_time<<" s"<<endl;
				mutex_unlock(&this->data_lock);
			}

            /**
             * @Synopsis GetChunk read a chunk from the buffer
             *
             * @Returns reference to a chunk of data
             */
			inline virtual ElemType& GetChunk() {
				mutex_lock(&this->data_lock);
				//check if there is available data
				if (this->rd_ptr->is_parsed == true){
					this->rd_ptr->is_parsed = false;
                    ElemType* p = this->rd_ptr;
					mutex_unlock(&this->data_lock);
					return *(p);
				}
				else { //no available data 
					if (this->is_on_loading == false){
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

            /**
             * @Synopsis FinishRead finished processing the read chunk
             */
			inline virtual void FinishRead() {
				mutex_lock(&this->data_lock);
				this->rd_ptr->is_inuse = false;
				//notice that the last data have been processed
				this->rd_ptr = this->rd_ptr->next;
				condition_variable_signal_all(&this->buffer_full);
				mutex_unlock(&this->data_lock);
			}
	};
}

#endif

