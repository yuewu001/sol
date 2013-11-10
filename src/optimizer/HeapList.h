/*************************************************************************
  > File Name: HeapList.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Sat 09 Nov 2013 11:03:03 AM
  > Descriptions: Heap list to select topK elements 
 ************************************************************************/

#ifndef HEADER_HEAP_LIST
#define HEADER_HEAP_LIST

#include "../common/global.h"
#include <assert.h>

#include <new>
#include <iostream>

using namespace std;
namespace SOL{
    template <typename T> class HeapList{
        public:
            IndexType *id2pos_map; //record the sorted position of each data
            IndexType *pos2id_map; //record the index of weight for each sorted position

            IndexType K; //keep top K elemetns
            IndexType data_num;

            const T* value_list;  //const pointer to the data

        public:
            HeapList():id2pos_map(NULL), pos2id_map(NULL)
                       , K(0), data_num(0), value_list(NULL){}

        private:
            /**
             * release :release memory
             */
            void release(){
                if (this->id2pos_map != NULL){
                    delete []this->id2pos_map;
                    this->id2pos_map = NULL;
                }
                if (this->pos2id_map != NULL){
                    delete []this->pos2id_map;
                    this->pos2id_map = NULL;
                }
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
            bool Init(IndexType dataNum, IndexType topK, const T* init_value){
                assert(topK > 0 && init_value != NULL);
                this->release();

                try{
                    this->pos2id_map = new IndexType[topK];
                    this->id2pos_map = new IndexType[dataNum];
                }catch(std::bad_alloc &ex){
                    std::cerr<<ex.what()<<"\nallocate memory for pos2id_map failed!"<<std::endl;
                    this->release();
                    return false;
                }
                //init all positions 
                for (IndexType i = 0; i < dataNum; i++){
                    this->id2pos_map[i] = i; 
                }
                for (IndexType i = 0; i < topK; i++){
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
            bool Init(IndexType dataNum, IndexType topK, 
                    const IndexType* init_id, const T* init_value){
                assert(topK > 0 && init_id != NULL && init_value != NULL);
                this->release();
                try{
                    this->pos2id_map = new IndexType[topK];
                    this->id2pos_map = new IndexType[dataNum];
                }catch(std::bad_alloc &ex){
                    std::cerr<<ex.what()<<"\nallocate memory for pos2id_map failed!"<<std::endl;
                    this->release();
                    return false;
                }
                //init all the pos of ids to topK
                for (IndexType i = 0; i != dataNum; i++){
                    this->id2pos_map[i] = topK;
                }

                for (IndexType i = 0; i != topK; i++){
                    this->pos2id_map[i] = init_id[i];
                }
                for (IndexType i = 0; i != topK; i++){
                    this->id2pos_map[this->pos2id_map[i]] = i;
                }

                this->K = topK;
                this->data_num = dataNum;
                this->value_list = init_value; 
                this->BuildHeap();
                return true;
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
            bool UpdateHeap(IndexType data_id, IndexType &ret_id){
                IndexType cur_pos = this->id2pos_map[data_id];
                if (cur_pos < this->K){
                    IndexType par_pos = (cur_pos - 1) / 2;
                    if (cur_pos == 0)
                        par_pos = 0;
                    //the current value increases
                   if(this->value_list[this->pos2id_map[cur_pos]] > 
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
                           par_pos = (cur_pos - 1)/ 2;
                       }while(par_pos >= 0 && this->value_list[this->pos2id_map[cur_pos]] > 
                            this->value_list[this->pos2id_map[par_pos]]);
                       
                       return false;
                    }
                   //the current value decreases
                   this->HeapAdjust(cur_pos, this->K - 1);
                   return false;
                }
                else{
                    ret_id = this->pos2id_map[0];
                    T thresh_val = this->value_list[ret_id];
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
                    IndexType top_id =this->pos2id_map[0];
                    this->pos2id_map[0] = this->pos2id_map[i];
                    this->id2pos_map[this->pos2id_map[i]] =  0; 
                    this->pos2id_map[i] = top_id;
                    this->id2pos_map[top_id] = i;

                    this->HeapAdjust(0, i - 1);
                }
            }

            void Output(){
                cout<<"pos to id ";
                for (IndexType i = 0; i != this->K; i++){
                    std::cout<<this->pos2id_map[i]<<" ";
                }
                std::cout<<"\nid to pos ";
                for (IndexType i = 0; i != this->K; i++){
                    std::cout<<this->id2pos_map[this->pos2id_map[i]]<<" ";
                    //std::cout<<this->value_list[this->pos2id_map[i]]<<" ";
                }
                std::cout<<"\n";

            }

        private:
            void BuildHeap(){
                IndexType i = (this->K - 1) / 2 + 1;
                do{
                    i--;
                    HeapAdjust(i, this->K - 1);
                }while(i > 0);
            }

            /**
             * HeapAdjust : adjust the heap to satisfy heap properties
             *
             * @Param s: r[s+1, ..., m] is a heap, adjust the heap so that r[s,..,m] is a heap
             * @Param m
             */
            void HeapAdjust(IndexType s, IndexType m){
                if (s >= this->K){
                    cout<<s<<endl;
                }
                assert(s >= 0 && s < this->K);
                IndexType cur_id = this->pos2id_map[s];
                T cur_val = this->value_list[cur_id];

                IndexType i = s;
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
            void UpdateDataNum(IndexType new_num, const T* value_list){
                if (new_num < data_num || this->id2pos_map == NULL)
                    return;

                IndexType *newFM = new IndexType[new_num];
                //copy info
                memcpy(newFM, this->id2pos_map, sizeof(IndexType)* this->data_num);
                //set the rest
                for (IndexType i = this->data_num; i < new_num; i++)
                    newFM[i] = i;
                delete []this->id2pos_map;
                this->id2pos_map = newFM;
                this->data_num = new_num;

                this->value_list = value_list;
            }
    };
}

#endif

