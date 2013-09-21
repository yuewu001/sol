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

void TestIO_Cpp(const string &fileName)
{
    ifstream inFile(fileName.c_str(),ios::in);
    string line;
    int lineNum = 0;
    size_t len = 0;
    while(inFile.good())
    {
        getline(inFile,line);
        lineNum++;
        len += line.length();
    }
    inFile.close();
    cout<<"Read "<<lineNum<<" lines!"<<endl;
    cout<<"total line "<<len<<endl;
}

void TestIO_C(const char* fileName)
{
    FILE *fp = fopen(fileName,"r");
    if(fp == NULL)
    {
        printf("Open file failed!\n");
        return;
    }
    int lineNum = 0;
    int maxLen = 40960;
    char* line = new char[maxLen];
    while(fgets(line,maxLen,fp) != NULL)
        lineNum++;

    fclose(fp);
    printf("line number: %d\n",lineNum);
}

int main(int argc, char** args)
{
    string fileName = "../data/rcv1.train";

    /*
    clock_t time1 = clock();
    TestIO_C(fileName.c_str());
    clock_t time2 = clock();
    cout<<"time elapsed for c "<<(float)(time2 - time1)/CLOCKS_PER_SEC * 1000 <<" ms"<<endl;
    TestIO_Cpp(fileName);
    clock_t time3 = clock();
    cout<<"time elapsed for c++ "<<(float)(time3 - time2)/CLOCKS_PER_SEC * 1000 <<" ms"<<endl;
    return 0;
    */

    clock_t start = clock();
    LibSVMReader reader(fileName);
    DataSet<float,char> dataset(1,5000);

    dataset.Load(reader);
    clock_t end = clock();
    cout<<"time elapsed: "<<(float)(end - start) / CLOCKS_PER_SEC<<" s"<<endl;

    return 0;
}
