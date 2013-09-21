/*************************************************************************
	> File Name: test.cpp
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2013/9/20 13:18:02
	> Functions: 
 ************************************************************************/

#include "DataSet.h"
#include "libsvmread.h"

#include <string>
#include <iostream>
#include <ctime>

using namespace std;
using namespace SOL;

int main(int argc, char** args)
{
    string fileName = "../SOL/rcv1.train";

    clock_t start = clock();
    LibSVMReader reader(fileName);
    DataSet<float,char> dataset(1,5000);

    dataset.Load(reader);
    clock_t end = clock();
    cout<<"time elapsed: "<<(float)(end - start) / CLOCKS_PER_SEC<<" ms"<<endl;

    return 0;
}
