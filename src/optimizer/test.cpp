#include "HeapList.h"

#include <iostream>
#include <time.h>
#include <cstdlib>

using namespace std;
using namespace SOL;

int main(int argc, char** args){
    int len = 20;
    float* val_list = new float[len];
    for(int i = 0; i < len; i++){
        val_list[i] = (rand() %100)/ 100.0;
    }
    HeapList<float> heap;
    heap.Init(len, len / 2, val_list);
    heap.Output();
//    heap.HeapSort();
    heap.Output();

    for (int i = 8; i < 20; i++){
        cout<<"\n";
        heap.Output();
        val_list[i] = (rand() %100)/ 10.0;
        cout<<i<<"\t";
        cout<<val_list[i]<<endl;
        heap.UpdateHeap(i);
        heap.Output();
    }
    //heap.Output();

    return 0;
}
