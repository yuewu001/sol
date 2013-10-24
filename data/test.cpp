/*************************************************************************
  > File Name: test.cpp
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Wed 23 Oct 2013 02:05:35 PM
  > Descriptions: 
 ************************************************************************/

#include "libsvmread.h"

#include <iostream>
#include <ctime>
#include <string>

using namespace std;
using namespace SOL;

int main(int argc, char**args){
    string filename = "/home/matthew/work/Data/rcv1/rcv1.train";
    //string filename = "temp_data";

    clock_t start = clock();;

    LibSVMReader reader(filename);
    if (reader.OpenReading() == false){
        return -1;
    }
    int datanum = 0;

    DataPoint<float,char> point;
    for (int i = 0; i < 1; i++){
        reader.Rewind();
        while(reader.GetNextData(point) == true){
            datanum++;
        }
    }
    clock_t end = clock();
    cout<<"data number: "<<datanum<<endl;
    cout<<"time elapsed: "<<(end - start) * 1000.0 / CLOCKS_PER_SEC<<" ms"<<endl;
    return 0;
}
