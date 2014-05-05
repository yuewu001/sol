/*************************************************************************
  > File Name: MaxHeap.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Sat 09 Nov 2013 11:03:03 AM
  > Descriptions: Heap list to select topK elements
  ************************************************************************/

#ifndef HEADER_MAX_HEAP_LIST
#define HEADER_MAX_HEAP_LIST

#include "s_array.h"
#include "config.h"

#include <new>
#include <iostream>

using namespace std;
namespace BOC{
	template <typename value_type, typename index_type = IndexType> class MaxHeap{
	private:
		s_array<index_type> id2pos_map; //record the sorted position of each data
		s_array<index_type> pos2id_map; //record the index of weight for each sorted position

		index_type K; //keep top K elemetns
		index_type data_num;

		const value_type* value_list;  //const pointer to the data

	public:
		MaxHeap() :K(0), data_num(0), value_list(NULL){}
		index_type GetK() const { return this->K;}

	public:
		inline index_type get_id(index_type pos) const {
			return this->pos2id_map[pos];
		}
		inline index_type get_pos(index_type id) const {
			return this->id2pos_map[id];
		}
		inline bool is_topK(index_type id) const {
			return this->id2pos_map[id] < this->K;
		}
		inline value_type GetHeapLimit() const {
			return this->value_list[this->pos2id_map[0]];
		}

	private:
		/**
		 * release :release memory
		 */
		void release(){
			this->pos2id_map.resize(0);
			this->id2pos_map.resize(0);
			this->K = 0;
			this->data_num = 0;
			this->value_list = NULL;
		}

	public:
		/**
		 * Init : init the heap with provided initial value
		 *
		 * @Param dataNum: data number in total
		 * @Param topK: keep top K elements in the heap
		 * @Param init_value: initial values
		 */
		bool Init(index_type dataNum, index_type topK, const value_type* init_value){
			//assert(topK > 0 && init_value != NULL);
			if (topK < 1){
				std::cerr << "ERROR: topK should be no less than 1, current value " <<topK<< std::endl;
				exit(-1);
			}

			this->release();
			this->pos2id_map.resize(topK);
			this->id2pos_map.resize(dataNum);

			//init all positions 
			for (index_type i = 0; i < dataNum; i++){
				this->id2pos_map[i] = i;
			}
			for (index_type i = 0; i < topK; i++){
				this->pos2id_map[i] = i;
			}

			this->K = topK;
			this->data_num = dataNum;
			this->value_list = init_value;
			this->BuildHeap();
			return true;
		}

		/**
		 * Init : init the heap with provided initial id and initial value
		 *
		 * @Param dataNum: data number in total
		 * @Param init_id: initial data ids
		 * @Param topK: keep top K elements in the heap
		 * @Param init_value: initial values
		 */
		bool Init(index_type dataNum, index_type topK,
			const index_type* init_id, const value_type* init_value){
			//assert(topK > 0 && init_id != NULL && init_value != NULL);
			this->release();
			this->pos2id_map.resize(topK);
			this->id2pos_map.resize(dataNum);

			//init all the pos of ids to topK
			for (index_type i = 0; i != dataNum; i++){
				this->id2pos_map[i] = topK;
			}

			for (index_type i = 0; i != topK; i++){
				this->pos2id_map[i] = init_id[i];
			}
			for (index_type i = 0; i != topK; i++){
				this->id2pos_map[this->pos2id_map[i]] = i;
			}

			this->K = topK;
			this->data_num = dataNum;
			this->value_list = init_value;
			this->BuildHeap();
			return true;
		}

	public:
		void ResizeHeap(index_type newK){
			if (newK == this->K)
				return;
			else if (newK > this->K)
				this->EnlargeHeap(newK);
			else
				this->ShrinkHeap(newK);
		}

	protected:
		/**
		*	Enlarge number of K in the heap
		**/
		void EnlargeHeap(index_type newK){
			if (newK < this->K){
				cerr << "new K(" << newK << ") must be no less than original K(" << this->K<< ") in EnlargeHeap!" << endl;
			}
			else if (newK == this->K)
				return;

			this->pos2id_map.reserve(newK);
			this->pos2id_map.resize(newK);
			//update id2pos for those ids not in the heap
			//update pos2id fr new added pos
			for (index_type i = 0, j = this->K; i != this->data_num; i++){
				if (this->is_topK(i) == false){
					if (j < newK){
						this->pos2id_map[j] = i;
						this->id2pos_map[i] = j;
						j++;
					}
					else
						this->id2pos_map[i] = newK;
				}
			}
			this->K = newK;
			this->BuildHeap();
			//make sure most confident values are in the heap
			index_type ret_id;
			for (index_type i = 0; i != this->data_num; i++){
				if (this->is_topK(i) == false){
					this->UpdateHeap(i, ret_id);
				}
			}
		}

		void ShrinkHeap(index_type newK){
			if (newK > this->K)
				cerr << "new K(" << newK << ") must be no larger than original K(" << this->K<< ") in ShrinkHeap!" << endl;
			else if (newK == this->K)
				return;
			else{
				this->pos2id_map.resize(newK);
				this->K = newK;
				this->BuildHeap();
				//make sure most confident values are in the heap
				index_type ret_id;
				for (index_type i = 0; i != this->data_num; i++){
					if (this->is_topK(i) == false){
						this->UpdateHeap(i, ret_id);
					}
				}
			}
		}

	public:

		/**
		 * UpdateHeap : update the value of the specified item and adjust heap
		 * Note that: when value in the heap increases to the largest among all
		 * the data, we donot move it out and move the outsite smallest in
		 *
		 * @Param data_id: specified data item
		 * @Param value: new value of the item

		 * @Return: true if replaced by the input data_id
		 */
		bool UpdateHeap(index_type data_id, index_type &ret_id){
			index_type cur_pos = this->id2pos_map[data_id];
			if (cur_pos < this->K){
				index_type par_pos = (cur_pos - 1) / 2;
				if (cur_pos == 0)
					par_pos = 0;
				//the current value increases
				if (this->value_list[this->pos2id_map[cur_pos]] >
					this->value_list[this->pos2id_map[par_pos]]){
					do{
						//swap parent and itself
						this->pos2id_map[cur_pos] = this->pos2id_map[par_pos];
						this->id2pos_map[this->pos2id_map[par_pos]] = cur_pos;

						this->pos2id_map[par_pos] = data_id;
						this->id2pos_map[data_id] = par_pos;

						cur_pos = par_pos;
						if (cur_pos == 0)
							break;
						par_pos = (cur_pos - 1) / 2;
					} while (par_pos >= 0 && this->value_list[this->pos2id_map[cur_pos]] >
						this->value_list[this->pos2id_map[par_pos]]);

					return false;
				}
				//the current value decreases
				this->HeapAdjust(cur_pos, this->K - 1);
				return false;
			}
			else{
				ret_id = this->pos2id_map[0];
				value_type thresh_val = this->value_list[ret_id];
				if (this->value_list[data_id] < thresh_val){
					//swap with the top element of the heap
					this->id2pos_map[ret_id] = this->K;
					this->id2pos_map[data_id] = 0;
					this->pos2id_map[0] = data_id;
					this->HeapAdjust(0, this->K - 1);

					return true;
				}
				else{
					ret_id = data_id;
					return true;
				}
			}
		}

	public:
		void HeapSort(){
			this->BuildHeap();
			for (int i = this->K - 1; i > 0; i--){
				//swap top and last
				index_type top_id = this->pos2id_map[0];
				this->pos2id_map[0] = this->pos2id_map[i];
				this->id2pos_map[this->pos2id_map[i]] = 0;
				this->pos2id_map[i] = top_id;
				this->id2pos_map[top_id] = i;

				cout << this->value_list[top_id] << " ";
				this->HeapAdjust(0, i - 1);
			}
			cout << endl;
		}

		void Output(){
			cout << "pos to id ";
			for (index_type i = 0; i != this->K; i++){
				std::cout << this->pos2id_map[i] << " ";
			}
			std::cout << "\nid to pos ";
			for (index_type i = 0; i != this->K; i++){
				std::cout << this->id2pos_map[i] << " ";
				//std::cout<<this->value_list[this->pos2id_map[i]]<<" ";
			}
			std::cout << "\n";

		}

	private:
		void BuildHeap(){
			index_type i = (this->K - 1) / 2 + 1;
			do{
				i--;
				HeapAdjust(i, this->K - 1);
			} while (i > 0);
		}

		/**
		 * HeapAdjust : adjust the heap to satisfy heap properties
		 *
		 * @Param s: r[s+1, ..., m] is a heap, adjust the heap so that r[s,..,m] is a heap
		 * @Param m
		 */
		void HeapAdjust(index_type s, index_type m){
			//assert(s >= 0 && s < this->K);
			index_type cur_id = this->pos2id_map[s];
			value_type cur_val = this->value_list[cur_id];

			index_type i = s;
			for (i = 2 * s + 1; i <= m; i = 2 * i + 1){
				if (i < m &&
					this->value_list[this->pos2id_map[i]] <
					this->value_list[this->pos2id_map[i + 1]])
					i++; //j is the bigger child

				if (cur_val >= this->value_list[this->pos2id_map[i]])
					break;
				//set parent node to be child node
				this->pos2id_map[s] = this->pos2id_map[i];
				this->id2pos_map[this->pos2id_map[i]] = s;
				s = i;
			}

			//insert
			this->pos2id_map[s] = cur_id;
			this->id2pos_map[cur_id] = s;
		}

	public:
		/**
		 * UpdateDataNum : increase the total data number
		 *
		 * @Param new_num: new total data number
		 */
		void UpdateDataNum(index_type new_num, const value_type* value_list){
			if (new_num < data_num)
				return;

			this->id2pos_map.resize(new_num);
			//set the rest
			for (index_type i = this->data_num; i < new_num; i++)
				this->id2pos_map[i] = i;
			this->data_num = new_num;

			this->value_list = value_list;
		}
	};
}

#endif

