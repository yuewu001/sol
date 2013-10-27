/*************************************************************************
  > File Name: s_array.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: 2013/9/19 15:14:53
  > Functions: customized array
 ************************************************************************/

#pragma once 

#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <cstring>

namespace SOL
{
    template <typename T> class s_array
    {
        public:
            T* begin; //point to the first element
            T* end; //point to the next postion of the last element
            size_t capacity; //capacity of the array
            int *count;

            T first() const {return *begin;}
            T last() const {return *(end - 1);}
            T pop() {return *(--end);}
            bool empty() const {return begin == end;}
            size_t size() const {return end - begin;}
            T& operator[] (size_t i) {return begin[i];}
            const T& operator[] (size_t i) const { return begin[i];}

            void resize(size_t newSize, bool isClear = false)
            {
                if (capacity >= newSize) //just need to modify the pointer, if resize the array smaller
                {
                    end = begin + newSize;
                    if (isClear == true)
                        memset(begin,0,sizeof(T) * newSize);
                }
                else if (capacity < newSize) //allocate more memory
                {
                    T* new_begin = new T[newSize];
                    if (new_begin == NULL && sizeof(T) * newSize > 0)
                    {
                        std::cerr<<"realloc of "<<newSize<<" failed in resize(). out of memory?\n" 
                            <<__FILE__<<"\n"<<__LINE__<<std::endl;
                        throw std::exception();
                    }

                    size_t old_len = this->size();
                    if (isClear == false) //copy data
                        memcpy(new_begin,begin,sizeof(T) * old_len);
                    if (begin != NULL)
                        delete []begin;
                    begin = new_begin;
                    end = begin + newSize;
                    capacity = newSize;
                }
            }
            void erase(void) { resize(0); }

            void push_back(const T& elem)
            {
                size_t old_len = size();
                if (old_len == capacity) //full array
                {
                    resize(2 * old_len + 3);
                    end = begin + old_len;
                }
                *(end++) = elem;
            }

            s_array<T>& operator= (const s_array<T> &arr)
            {
                if (this->count == arr.count)
                    return *this;
                this->release();

                this->begin =arr.begin;
                this->end = arr.end;
                this->capacity = arr.capacity;
                this->count = arr.count;
                ++(*count);
                return *this;
            }

            void release()
            {
                --(*count);
                if (*count == 0)
                {
                    if (this->begin != NULL)
                        delete []this->begin;
                    delete this->count;
                }
                this->begin = NULL;
                this->end = NULL;
                this->capacity = 0;
                this->count = 0;
            }

            s_array()
            {
                begin = NULL; end = NULL; capacity = 0;
                count = new int;
                *count = 1;
            }
            s_array(const s_array &arr)
            {
                this->begin =arr.begin;
                this->end = arr.end;
                this->capacity = arr.capacity;
                this->count = arr.count;
                ++(*count);
            }

            ~s_array() { this->release(); }
    };
}
