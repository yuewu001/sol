/*************************************************************************
  > File Name: test.cpp
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Mon 04 Nov 2013 09:50:06 PM
  > Descriptions: 
 ************************************************************************/
#if defined(_MSC_VER) && defined(_DEBUG)
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif

#include "Params.h"

#include "Converter.h"

#include "../io/libsvm_io.h"
#include "../io/libsvm_binary.h"
#include "../io/DataSet.h"


#include <fstream>
#include <cstdio>
#include <vector>

using namespace SOL;

int main(int argc, const char** args){
	//check memory leak in VC++
#if defined(_MSC_VER) && defined(_DEBUG)
	int tmpFlag = _CrtSetDbgFlag( _CRTDBG_REPORT_FLAG );
	tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
	_CrtSetDbgFlag( tmpFlag );
	//_CrtSetBreakAlloc(1698);  
#endif

    Params param;
    if (param.Parse(argc,args) == false)
        return -1;

	if (param.is_de_cache == true){
		param.str_data_type = "cache";
		Convert(param);
	}
	else if (param.is_cache == true)
		Cache(param);
	return 0;
}

void Cache(const Params &param){
	cout<<"Caching file..."<<endl;
	
	DataSet<float, char> dt;
	DataReader<float, char> *reader = getReader<float, char>(param.in_fileName, param.str_data_type);
	if (reader == NULL){
		return;
	}

	dt.Load(reader,param.out_fileName);
	size_t dataNum = 0;

	size_t show_step = 1; //show information every show_step
	size_t show_count = 2;
	if(dt.Rewind()){
		while(1){
			const DataChunk<float, char> chunk = dt.GetChunk();
			dataNum += chunk.dataNum;
			if (chunk.dataNum == 0){
				dt.FinishRead();
				break;
			}
			dt.FinishRead();
			if (show_count < dataNum){
				printf("%lu samples cached\r",dataNum);
				show_count = (1 << ++show_step);
			}
		}
	}
	printf("%lu samples cached\n",dataNum);
	if (reader != NULL)
		delete reader;
}

void Convert(const Params &param){
	cout<<"Convert file to libsvm"<<endl;
	DataReader<float, char> *reader = getReader<float, char>(param.in_fileName,param.str_data_type);
	if (reader->OpenReading() == false){
		cerr<<"open "<<param.in_fileName<<" failed!"<<endl;
		return;
	}
	string tmp_filename = param.out_fileName + ".writing";

	libsvm_io writer(tmp_filename);
	if(writer.OpenWriting() == false){
		cerr<<"open output file" <<tmp_filename<<" failed!"<<endl;
		return;
	}
	DataPoint<float, char> data;
	size_t dataNum = 0;
	size_t featNum = 0;
	size_t show_step = 1; //show information every show_step
	size_t show_count = 2;
	while(reader->GetNextData(data) == true){
		dataNum++;
		featNum += data.indexes.size();

		if (writer.WriteData(data) == false){
			break;
		}

		if (show_count < dataNum){
			printf("%lu samples de-cached\r",dataNum);
			show_count = (1 << ++show_step);
		}
	}
	writer.Close();
	if (reader->Good() == true && 
		rename_file(tmp_filename, param.out_fileName) == true)
		printf("%lu samples (%lu features) de-cached\n",dataNum, featNum);
	reader->Close();
	delete reader;
}
