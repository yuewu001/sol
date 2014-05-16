/*************************************************************************
  > File Name: MPDataSet.h
  > Copyright (C) 2014 Yue Wu<yuewu@outlook.com>
  > Created Time: 2014/2/9 Saturday 15:38:09
  > Functions: Class to interact with datasets with multi pass
  ************************************************************************/

#ifndef HEADER_MP_DATASET
#define HEADER_MP_DATASET

#include "OnlineBuffer.h"
#include "MPChunk.h"

#include "../utils/util.h"

/**
 *  namespace: Sparse Online Learning
 */
namespace BOC {
	//data set, can work in both read-and-write mode and read-once mode
	template <typename FeatType, typename LabelType>
	class OnlineMPBuffer : public OnlineBuffer<FeatType, LabelType>{

        DECLARE_CLASS

	protected:
		//added by yuewu, for multi-pass
		MPBuffer<FeatType, LabelType> *pMp_Buffer;

	public:
		OnlineMPBuffer(int buf_size, int chunk_size)
			: OnlineBuffer<FeatType, LabelType>(buf_size, chunk_size){
				this->pMp_Buffer = NULL;
			}

		~OnlineMPBuffer() {
			DELETE_POINTER(this->pMp_Buffer);
		}

		//get the data to read
		virtual ChunkType& GetChunk() {
			mutex_lock(&this->data_lock);
			//check if there is available data
			if (this->rd_ptr->is_parsed == true){
				this->rd_ptr->is_parsed = false;
				mutex_unlock(&this->data_lock);
				return *(this->rd_ptr);
			}
			else{ //no available data 
				if (this->is_on_loading == false){
						this->rd_ptr->is_parsed = false;
						this->rd_ptr->erase();
						mutex_unlock(&this->data_lock);
						return *(this->rd_ptr); //return an invalid data
				}
				else{ //suspend the current thread
					if (this->pMp_Buffer == NULL || this->pMp_Buffer->dataNum == 0){
						condition_variable_wait(&this->data_available, &this->data_lock);
						mutex_unlock(&this->data_lock);
						return this->GetChunk();
					}
					else{
						this->pMp_Buffer->is_inuse = true;
						mutex_unlock(&this->data_lock);
						return *this->pMp_Buffer;
					}
				}
			}
		}

		virtual void FinishRead() {
			if (this->pMp_Buffer == NULL){
				OnlineDataSet<FeatType, LabelType>::FinishRead();
				return;
			}
			else{
				mutex_lock(&this->data_lock);

				if (this->pMp_Buffer->is_inuse == false){
					//check if the next chunk is ready
					//if (this->rd_ptr->next->is_parsed == false){
					{
						//not ready, clone into mp buffer
						for (size_t i = 0; i < this->rd_ptr->dataNum; i++){
							this->pMp_Buffer->Push(this->rd_ptr->data[i]);
						}
					}

					this->rd_ptr->is_inuse = false;
					//notice that the last data have been processed
					this->rd_ptr = this->rd_ptr->next;
					condition_variable_signal_all(&this->buffer_full);
				}
				else{
					this->pMp_Buffer->is_inuse = false;
				}

				mutex_unlock(&this->data_lock);
			}
		}
	};

	IMPLEMENT_DATA_CLASS(OnlineMPBuffer, "OnlineMP")
}
#endif
