/*************************************************************************
	> File Name: main.cpp
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2013/9/20 13:18:02
	> Functions:
	************************************************************************/
#include "BOC.h"

#include <string>
#include <iostream>
#include <fstream>

using namespace std;
using namespace BOC;

#define FeatType float
#define LabelType char

int main(int argc, const char** args) {
	//check memory leak in VC++
#if defined(_MSC_VER) && defined(_DEBUG)
	int tmpFlag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
	tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
	_CrtSetDbgFlag(tmpFlag);
	//_CrtSetBreakAlloc(170);
#endif

	Params param;
	if (param.Parse(argc, args) == false){
		return -1;
	}

	try{
		cout << param.StringValue("-i") << endl;
		cout << param.StringValue("-t") << endl;
		cout << param.IntValue("-passes") << endl;
	}
	catch (exception &ex){
		cerr << ex.what();
	}

	LibBOC<FeatType, LabelType> libBoc;
	libBoc.Initialize(param);

	return 0;
}
