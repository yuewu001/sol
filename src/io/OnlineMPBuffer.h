/*************************************************************************
	> File Name: OnlineMPBuffer.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 5/16/2014 2:59:23 PM
	> Functions: Online Buffer for multi-pass
	************************************************************************/
#ifndef HEADER_ONLINE_MP_BUFFER
#define HEADER_ONLINE_MP_BUFFER

#include "OnlineBuffer.h"
#include "MPChunk.h"

#include <stdexcept>

/**
 *  namespace: Batch and Online Classification
 */
namespace BOC {
	//data set, can work in both read-and-write mode and read-once mode
	template <typename PointType>
	class OnlineMPBuffer : public OnlineBuffer < PointType > {
	protected:
		MPChunk<PointType> *p_MPChunk;
		/**
		 * @Synopsis Constructors
		 */
	public:
		OnlineMPBuffer(int buf_size, int chunk_size) : OnlineBuffer<PointType>(buf_size, chunk_size) {
			this->p_MPChunk = NULL;
		}

		virtual ~OnlineMPBuffer() {
			DELETE_POINTER(this->p_MPChunk);
		}

	public:
		void ConfigMPBuffer(const string& mp_buf_type, int mp_buf_size) {
			DELETE_POINTER(this->p_MPChunk);
			if (mp_buf_type == "all"){
				this->p_MPChunk = new MPChunk_ALL<PointType>(mp_buf_size);
			}
			else if (mp_buf_type == "margin"){
				this->p_MPChunk = new MPChunk_LARGE_MARGIN<PointType>(mp_buf_size);
			}
			else if (mp_buf_type == "reservior"){
				this->p_MPChunk = new MPChunk_RESERVIOR<PointType>(mp_buf_size);
			}
			else{
				throw std::invalid_argument("Warnning: no buffer type for multi-pass is specified, no buffer for multi-pass is used!\n");
			}
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
		inline virtual ChunkType& GetChunk() {
			mutex_lock(&this->data_lock);
			//check if there is available data
			if (this->rd_ptr->is_parsed == true){
				this->rd_ptr->is_parsed = false;
				ChunkType * p = this->rd_ptr;
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
					if (this->p_MPChunk == NULL || this->p_MPChunk->dataNum == 0){
						condition_variable_wait(&this->data_available, &this->data_lock);
						mutex_unlock(&this->data_lock);
						return this->GetChunk();
					}
					else{
						this->p_MPChunk->is_inuse = true;
						mutex_unlock(&this->data_lock);
						return *this->p_MPChunk;
					}
				}
			}
		}

		/**
		 * @Synopsis FinishRead finished processing the read chunk
		 */
		inline virtual void FinishRead() {
			mutex_lock(&this->data_lock);
			if (this->p_MPChunk != NULL){
				if (this->p_MPChunk->is_inuse == true){
					this->p_MPChunk->is_inuse = false;
					mutex_unlock(&this->data_lock);
					return;
				}
				else{
					//check if the next chunk is ready
					//if (this->rd_ptr->next->is_parsed == false)
						{
							//not ready, clone into mp buffer
							for (size_t i = 0; i < this->rd_ptr->dataNum; i++){
								this->p_MPChunk->Push(this->rd_ptr->data[i]);
							}
						}
				}
			}
			this->rd_ptr->is_inuse = false;
			//notice that the last data have been processed
			this->rd_ptr = this->rd_ptr->next;
			condition_variable_signal_all(&this->buffer_full);
			mutex_unlock(&this->data_lock);
		}
	};
}

#endif

