
#ifndef _FULL_MAX_HEAP_H_
#define _FULL_MAX_HEAP_H_
#include<math.h>
#include<stdio.h>
#include "random.h"
#include "linear.h"
#include "shash.h"
#include "miniheap.h"
//#include<iostream.h>
//using namespace std;


template <class T>
class MaxHeap  
{
public:
	T* heapArray;
	int CurrentSize;
	int MaxSize;

public:
	MaxHeap(){};
	MaxHeap(const int n);
	void Allcate(int n);
	virtual ~MaxHeap()
	{delete []heapArray;};
	void BuildHeap();
	bool isLeaf(int pos) const;
	int leftchild(int pos) const;
	int rightchild(int pos) const;
	// Return parent position
	int parent(int pos) const;
	// 删除给定下标的元素
	bool Remove(int pos, T& node);
	bool Remove(int pos);
	void SiftDown(int left);
	//从position向上开始调整，使序列成为堆
	void SiftUp(int position); 
	bool Insert(T& newNode);
	T& DeleteMax();
	T Max() {return heapArray[0];}
	T GetElement(int i)
	{
		return  heapArray[i];;
	}
	void SetValue(int i, T elem);
	int Size() const {return CurrentSize;}
	bool empty()
	{
		if(CurrentSize>0)
			return false;
		else
			return true;
	}
};
 template<class T>
void MaxHeap<T>::SetValue(int i, T elem)
{
	//heapArray[i] = elem;
	Remove(i);
	Insert(elem);
}

template<class T>
MaxHeap<T>::MaxHeap(const int n)
{
	if(n<=0)
		return;
	CurrentSize=0;
	MaxSize=n;
	heapArray=new T[MaxSize];
	BuildHeap();
}
template<class T>
void MaxHeap<T>::Allcate(int n)
{
	if(n<=0)
		return;
	CurrentSize=0;
	MaxSize=n;
	heapArray=new T[MaxSize];
	BuildHeap();
}
template<class T>
void MaxHeap<T>::BuildHeap()
{
	for (int i=CurrentSize/2-1; i>=0; i--) 
		SiftDown(i); 
}

template<class T>
bool MaxHeap<T>::isLeaf(int pos) const
{
	return (pos>=CurrentSize/2)&&(pos<CurrentSize);
}

template<class T>
int MaxHeap<T>::leftchild(int pos) const
{
	return 2*pos+1;      //返回左孩子位置
}

template<class T>
int MaxHeap<T>::rightchild(int pos) const
{
	return 2*pos+2;      //返回右孩子位置
}

template<class T>
int MaxHeap<T>::parent(int pos) const // 返回父节点位置
{
	return (pos-1)/2;
}

template<class T>
void MaxHeap<T>::SiftDown(int left)
{
	//准备
	int i=left;       //标识父结点
	int j=2*i+1;      //标识关键值较小的子结点  
	T temp=heapArray[i];    //保存父结点
	//过筛
	while(j<CurrentSize)
	{
		if((j<CurrentSize-1)&&(heapArray[j]<heapArray[j+1]))
			j++;      //j指向右子结点
		if(temp<heapArray[j])
		{
			heapArray[i]=heapArray[j];
			i=j;
			j=2*j+1;
		}
		else break;
	}
	heapArray[i]=temp;
}

template<class T>
void MaxHeap<T>::SiftUp(int position) 
{//从position向上开始调整，使序列成为堆
	int temppos=position;
	T temp=heapArray[temppos];
	while((temppos>0)&&(heapArray[parent(temppos)]<temp))
	{
		heapArray[temppos]=heapArray[parent(temppos)];
		temppos=parent(temppos);
	}
	heapArray[temppos]=temp;
}

template<class T>
bool MaxHeap<T>::Insert( T& newNode)
{
	if(CurrentSize==MaxSize)
		return false;
	heapArray[CurrentSize] = newNode;
	SiftUp(CurrentSize);
	CurrentSize++;
	return true;
}

template<class T>
T& MaxHeap<T>::DeleteMax()
{
	if(CurrentSize==0)
	{
		printf("Can't Delete\n");

	}
	else
	{
		T temp=heapArray[0];     //取堆顶元素
		heapArray[0]=heapArray[CurrentSize-1]; //堆末元素上升至堆顶
		CurrentSize--;
		if(CurrentSize>1)
			SiftDown(0);      //从堆顶开始筛选
		return temp;
	}
}

template<class T>
bool MaxHeap<T>::Remove(int pos, T& node)
{// 删除给定下标的元素
	if((pos<0)||(pos>=CurrentSize))
		return false;
	T temp=heapArray[pos];
	heapArray[pos]=heapArray[--CurrentSize]; //指定元素置于最后
	SiftUp(pos);        //上升筛
	SiftDown(pos);        //向下筛 
	node=temp;
	return true;
}

template<class T>
bool MaxHeap<T>::Remove(int pos)
{// 删除给定下标的元素
	if((pos<0)||(pos>=CurrentSize))
		return false;
	T temp=heapArray[pos];
	heapArray[pos]=heapArray[--CurrentSize]; //指定元素置于最后
	SiftUp(pos);        //上升筛
	SiftDown(pos);        //向下筛 
	return true;
}


//基本思想是，取前面K个元素组成一个最大堆，后面的组成一个最小堆
class min_tree
{
public:
	min_tree(){};
	min_tree(int n);
	~min_tree()
	{
	}
	int insert(int elem_pos, weight elem);
	int change_elem(int elem_pos, weight elem);

	weight get_change_elem(int &elem_pos);
	weight get_elem(int elem_pos);
	int add_update(stdext::hash_map<size_t,double>   element_hash_map, int probn);
    MaxHeap <weight> Max_K_Heap;
	MinHeap <weight> Min_K_Heap;
protected:
private:
	//weight *half_array;
	int median_pos;
	double median_val;
	weight median_elem;
	int n_size;
	int current_size;

};

weight min_tree::get_elem(int elem_pos)
{
	weight weight_temp;
	if (elem_pos<median_pos)
	{
		weight_temp = Max_K_Heap.GetElement(elem_pos);
		return weight_temp;
	}else
	{
		weight_temp = Min_K_Heap.GetElement(elem_pos-median_pos);
		return weight_temp;
	}
}

min_tree::min_tree(int n)
{
	if(n<=0)
		return;
	n_size = n;
	median_pos = 400;
	median_val = 0.0;
	current_size = 0;
	//Max_K_Heap(n);
	//Min_K_Heap(n);
	Max_K_Heap.Allcate(median_pos+1);
	Min_K_Heap.Allcate(n-median_pos+1) ;
}

int min_tree::change_elem(int elem_pos, weight elem)
{
	weight weight_min = Min_K_Heap.Min();
	if (weight_min>elem)
	{
		Max_K_Heap.Remove(elem_pos);
		Max_K_Heap.Insert(elem);
	}
	else
	{
		//weight weight_max = Max_K_Heap->Max();
		Min_K_Heap.DeleteMin();
		Min_K_Heap.Insert(elem);

		//
		Max_K_Heap.Remove(elem_pos);
        Max_K_Heap.Insert(weight_min);
	}
	return 0;
}

weight min_tree::get_change_elem(int &elem_pos)
{
	//make sure median_pos-1>0
	weight weight_temp;
	int key_elem = random() %(median_pos-1);
	elem_pos = key_elem;
	weight_temp = Max_K_Heap.GetElement(key_elem);
	return weight_temp;
}



int min_tree::insert(int elem_pos, weight elem)
{
    weight max_weight;
	if (current_size < median_pos)
	{
		Max_K_Heap.Insert(elem);	
	}
	else
	{
		max_weight = Max_K_Heap.Max();
		if (max_weight>elem)
		{
			Max_K_Heap.DeleteMax();
			Max_K_Heap.Insert(elem);
			Min_K_Heap.Insert(max_weight);
		}
		else
		{
			Min_K_Heap.Insert(elem);
		}
		
	}
	current_size++;
	
	return 0;
}

int min_tree::add_update(stdext::hash_map<size_t,double>   element_hash_map, int probn)
{

	int select_idx;
	double w_value;

	weight w_max;
	weight w_min;
	weight weght_temp;

	//check from the first element
	//for (int i=0; i<median_pos; i++ )
	//{
	//	weght_temp = Max_K_Heap.GetElement(i);
	//	if (weght_temp.index2==-1)
	//	{
	//		select_idx = weght_temp.index1;
	//	}else
	//	{
	//		select_idx = (weght_temp.index1+1)*probn 
	//			- weght_temp.index1*(weght_temp.index1+1)/2 + weght_temp.index2;
	//	}

	//	//got the updated value
	//	w_value = element_hash_map[select_idx];

	//	//check whether updated
	//	if(abs(abs(w_value)-abs(weght_temp.value))>1e-6)
	//	{
	//		w_min = Min_K_Heap.Min();
	//		weght_temp.value = w_value;
	//		
	//		//check whether it is bigger than w_min 
	//		if (weght_temp>w_min)
	//		{
	//			//change
	//			Min_K_Heap.DeleteMin();
	//			Min_K_Heap.Insert(weght_temp);

	//			Max_K_Heap.Remove(i);
	//			Max_K_Heap.Insert(w_min);
	//		}
	//		else
	//		{
	//			Max_K_Heap.SetValue(i,weght_temp);
	//		}
	//		continue;
	//	}
	//}
	for (int i=0; i<current_size; i++ )
	{
		if (i<median_pos)
		{
			weght_temp = Max_K_Heap.GetElement(i);
			if (weght_temp.index2==-1)
			{
				select_idx = weght_temp.index1;
			}else
			{
				select_idx = (weght_temp.index1+1)*probn 
					- weght_temp.index1*(weght_temp.index1+1)/2 + weght_temp.index2;
			}
			w_value = element_hash_map[select_idx];
			if(abs(abs(w_value)-abs(weght_temp.value))>1e-6)
			{
				w_min = Min_K_Heap.Min();
				weght_temp.value = w_value;
				if (weght_temp>w_min)
				{
					//change
                    Min_K_Heap.DeleteMin();
					Min_K_Heap.Insert(weght_temp);
					Max_K_Heap.Remove(i);
					Max_K_Heap.Insert(w_min);
				}
				else
				{
					Max_K_Heap.SetValue(i,weght_temp);
				}
				continue;
			}
		}
		else
		{
		
			weght_temp = Min_K_Heap.GetElement(i-median_pos);
			if (weght_temp.index2==-1)
			{
				select_idx = weght_temp.index1;
			}else
			{
				select_idx = (weght_temp.index1+1)*probn 
					- weght_temp.index1*(weght_temp.index1+1)/2 + weght_temp.index2;
			}
			w_value = element_hash_map[select_idx];
			if(abs(abs(w_value)-abs(weght_temp.value))>1e-6)
			{
				w_max = Max_K_Heap.Max();
				weght_temp.value = w_value;
				if(weght_temp<w_max)
				{
					//change
					Max_K_Heap.DeleteMax();
					Max_K_Heap.Insert(weght_temp);

					Min_K_Heap.Remove(i-median_pos);
					Min_K_Heap.Insert(w_max);
				}
				else
				{
					Min_K_Heap.SetValue(i-median_pos,weght_temp);

				}
				continue;
			}
		}
		
	}
	
	return 0;
}

#endif