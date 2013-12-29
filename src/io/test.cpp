/*************************************************************************
	> File Name: test.cpp
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2013/11/22 19:30:27
	> Functions: 
 ************************************************************************/

#include "DataPoint.h"
#include <iostream>
using namespace SOL;
using namespace std;


struct A{
    DataPoint<float, char> pt;
};

int main(){
    DataPoint<float, char> pt;
    pt.AddNewFeat(1,1);
    pt.id = 0;

    A *a = new A;
    a->pt = pt.clone();

    delete a;

    return 0;
}

