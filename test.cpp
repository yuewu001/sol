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
#include <fstream>
#include <ctime>

using namespace std;
using namespace SOL;

void Test_Thread(DataSet<float,char> &dataset)
{
    int dataNum = 0;
    int featNum = 0;
//    dataset.BeginRead();
    while(1)
    {
        const DataPoint<float,char> &data = dataset.GetDataRd();
        if(data.indexes.size() == 0)
        {
            cout<<"all the data has been processed!"<<endl;
            break;
        }
        featNum += data.indexes.size();
        dataNum++;
        dataset.FinishRead();
    }
    cout<<"\nread data number: "<<dataNum<<endl;
    cout<<"data number : "<<dataset.size()<<endl;
}

int main(int argc, char** args)
{
    string fileName = "/home/matthew/SOL/data/rcv1.train";
    cout<<"Usage: test passNum bufSize"<<endl;
    if (argc == 1)
    {
        return 0;
    }
    int passNum = atoi(args[1]);
    int bufSize = 128;
    if (argc > 2)
        bufSize = atoi(args[2]);

    //LibSVMReader reader(fileName);
    DataSet<float,char> dataset(passNum,bufSize);

    clock_t start = clock();
    //dataset.Load(reader,"cache.sol");
    clock_t end = clock();
    //cout<<"time elapsed: "<<(float)(end - start) / CLOCKS_PER_SEC<<" s"<<endl;
    //cout<<"data number : "<<dataset.size()<<endl;

    start = clock();
    dataset.LoadCache("/home/matthew/SOL/code/cache.sol");
    Test_Thread(dataset);
    end = clock();
    cout<<"time elapsed: "<<(float)(end - start) / CLOCKS_PER_SEC<<" s"<<endl;

    return 0;
}
