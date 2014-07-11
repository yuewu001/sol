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

#include "../io/io_header.h"
#include "../utils/init_param.h"
#include "../utils/Params.h"

#include <fstream>
#include <cstdio>
#include <vector>

using namespace BOC;

void Convert(BOC::Params &param);
void Cache(BOC::Params &param);

void InitParms(Params& param);

int main(int argc, const char** args){
	//check memory leak in VC++
#if defined(_MSC_VER) && defined(_DEBUG)
	int tmpFlag = _CrtSetDbgFlag( _CRTDBG_REPORT_FLAG );
	tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
	_CrtSetDbgFlag( tmpFlag );
	//_CrtSetBreakAlloc(1698);  
#endif

	string ioInfo;
	IOInfo<float,char>::GetIOInfo(ioInfo);

    Params param;
	InitParms(param);
    if (param.Parse(argc,args) == false)
        return -1;
	string src_type = param.StringValue("-st");
	string dst_type = param.StringValue("-dt");
	ToLowerCase(src_type);
	ToLowerCase(dst_type);
	if (dst_type == "binary"){
		Cache(param);
	}
	else
		Convert(param);
	return 0;
}

void InitParms(Params& param){

	string overview = "Sparse Online Learning Library - Dataset Converter";
	string syntax = "Converter -i input_file -o output_file -st src_type -dt dst_type";
	string example = "Converter -i input_file -o output_file -st libsvm -dt csv";
	param.Init(overview, syntax, example);

	//input & output
	param.add_option("", 1, 1, "input file", "-i", " ");
	param.add_option("", 1, 1, "output file", "-o", " ");
	param.add_option("", 1, 1, "input dataset type", "-st", " ");
	param.add_option("", 1, 1, "output dataset type", "-dt", " ");
}

void Cache(Params &param){
	cout<<"Caching file..."<<endl;
	
	OnlineDataSet<float, char> dt(1, false,init_buf_size, init_chunk_size);
	string src_file = param.StringValue("-i");
    DataReader<float, char> *reader = (DataReader<float, char>*)Registry::CreateObject(param.StringValue("-st"), &src_file);
    if (reader == NULL){
		cerr << "create reader failed!" << endl;
        return;
    }

	dt.Load(reader,param.StringValue("-o"));
	size_t dataNum = 0;

	size_t show_step = 1; //show information every show_step
	size_t show_count = 2;
	dt.Rewind();
	while (1){
		const DataChunk<DataPoint<float, char> > chunk = dt.GetChunk();
		dataNum += chunk.dataNum;
		if (chunk.dataNum == 0){
			dt.FinishRead();
			break;
		}
		dt.FinishRead();
		if (show_count < dataNum){
			printf("%lu samples cached\r", dataNum);
			show_count = (size_t(1) << ++show_step);
		}
	}
	printf("%lu samples cached\n",dataNum);
	if (reader != NULL)
		delete reader;
}

template <typename FeatType, typename LabelType>
IndexType GetDataDim(DataReader<FeatType, LabelType> * reader){
	reader->Rewind();
	IndexType featDim = 0;
    DataPoint<FeatType, LabelType> data;
	while (reader->GetNextData(data) == true){
		if (featDim < data.dim())
			featDim = data.dim();
	}
	return featDim;
}

void Convert(Params &param){
	string src_type = param.StringValue("-st");
	string dst_type = param.StringValue("-dt");
	string in_file = param.StringValue("-i");
	string out_file = param.StringValue("-o");

	cout << "Convert file from "<<src_type<<" to " <<dst_type<< endl;
    DataReader<float, char> *reader = (DataReader<float, char>*)Registry::CreateObject(src_type, &in_file);
	if (reader == NULL){
		return;
	}

	string tmp_filename = out_file + ".writing";

	DataHandler<float, char> *writer = (DataHandler<float, char>*)Registry::CreateObject(dst_type, &tmp_filename);
	if (writer == NULL){
		return;
	}
	if (writer->OpenWriting() == false){
		cerr << "open output file" << tmp_filename << " failed!" << endl;
		return;
	}
	if (reader->OpenReading() == false){
		cerr << "open " << in_file << " failed!" << endl;
		return;
	}

	if (dst_type == "csv"){
		IndexType featDim = GetDataDim<float, char>(reader);
		if (writer->SetExtraInfo((const char*)(&featDim)) == false) {
			delete reader;
			delete writer;
			return;
		}
	}
	DataPoint<float, char> data;
	size_t dataNum = 0;
	size_t featNum = 0;
	size_t show_step = 1; //show information every show_step
	size_t show_count = 2;

	reader->Rewind();
	while (reader->GetNextData(data) == true){
		dataNum++;
		featNum += data.indexes.size();
		data.Sort();

		if (writer->WriteData(data) == false){
			break;
		}

		if (show_count < dataNum){
			printf("%lu samples converted\r", dataNum);
			show_count = (size_t(1) << ++show_step);
		}
	}
	writer->Close();
	if (reader->Good() == true &&
		rename_file(tmp_filename, out_file) == true)
		printf("%lu samples (%lu features) converted\n", dataNum, featNum);
	reader->Close();
	delete reader;
	delete writer;
}
