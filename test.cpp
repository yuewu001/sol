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
#include <cmath>

using namespace std;
using namespace SOL;

void Test_Thread(DataSet<float,char> &dataset)
{
    int dataNum = 0;
    int featNum = 0;
//    dataset.BeginRead();
    while(1)
    {
        const DataChunk<float,char> &chunk = dataset.GetChunk();
        if(chunk.dataNum  == 0)
        {
            cout<<"all the data has been processed!"<<endl;
            break;
        }
        float sum = 0;
        for (int i = 0; i < chunk.dataNum; i++)
        {
            const DataPoint<float,char> &data = chunk.data[i];
            int dim = data.indexes.size();
            for (int j = 0; j < dim; j++)
                sum = std::exp(data.features[j]);
        }

        dataNum += chunk.dataNum;
        dataset.FinishRead();
    }
    cout<<"\nread data number: "<<dataNum<<endl;
    cout<<"data number : "<<dataset.size()<<endl;
}

void Convert(const string &fileName)
{
    LibSVMReader reader(fileName);
    DataSet<float,char> dataset;

    clock_t start = clock();
    dataset.Load(reader,"cache.sol");
    clock_t end = clock();
    cout<<"time elapsed: "<<(float)(end - start) / CLOCKS_PER_SEC<<" s"<<endl;
    cout<<"data number : "<<dataset.size()<<endl;
}

int main(int argc, char** args)
{
    string fileName = "/home/matthew/SOL/data/rcv1.train";
    cout<<"Usage: test passNum bufSize"<<endl;
    //int passNum = atoi(args[1]);
    int passNum = 1;
    int bufSize = 2;
//    bufSize = atoi(args[1]);
    
    DataSet<float,char> dataset(passNum,bufSize);
    clock_t start = clock();
    dataset.LoadCache("/home/matthew/SOL/data/cache.sol");
    Test_Thread(dataset);
    clock_t end = clock();
    cout<<"time elapsed: "<<(float)(end - start) / CLOCKS_PER_SEC<<" s"<<endl;
    cout<<"buffer size: "<<bufSize<<endl;

    return 0;
}
