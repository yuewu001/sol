/*************************************************************************
  > File Name: data_analysis.cpp
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Thu 24 Oct 2013 08:09:38 PM
  > Descriptions: analyse the sparsity of data
 ************************************************************************/

#include "../io/io_header.h"

#include "../utils/Params.h"

#include <string>
#include <map>

using namespace std;
using namespace BOC;

template <typename FeatType, typename LabelType>
bool Analyze(DataReader<FeatType, LabelType> *reader) {
    if (reader == NULL){
        cerr<<"data reader is empty!"<<endl;
        return false;
    }

    size_t max_show_count = 100000;
    size_t show_count = 1000;
    size_t dataNum = 0;
    size_t featNum = 0;
    IndexType max_index = 0;
    s_array<char> index_set;
	map<int, size_t> map_class_sample_num;

    DataPoint<FeatType, LabelType> data;
    if (reader->OpenReading() == true) {
        reader->Rewind();
        while(true) {
            if (reader->GetNextData(data) == true) {
                if (data.indexes.size() == 0)
                    continue;
                if (max_index < data.dim()){
                    max_index = data.dim();
                }
                size_t prev_size = index_set.size();
                if (max_index > prev_size){
                    index_set.resize(max_index);
                    //set the new value to zero
                    index_set.zeros(index_set.begin + prev_size, 
                            index_set.end);
                }
                for (size_t i = 0; i < data.indexes.size(); i++){
                    index_set[data.indexes[i] - 1] = 1;
                }

                dataNum++;

				int label = data.label;
				if (map_class_sample_num.find(label) != map_class_sample_num.end()){
					map_class_sample_num[label] += 1;
				}
				else{
					map_class_sample_num[label] = 1;
				}

                featNum += data.indexes.size();

                if (dataNum % show_count == 0){
                    cerr<<"data number  : "<<dataNum<<"    ";
                    cerr<<"valid dim    : "<<max_index<<"\r";
                    show_count *= 2;
                    show_count = show_count > max_show_count ? 
                        max_show_count : show_count;
                }
            }
            else
                break;
        }
    }
    else {
        cerr<<"Can not open file to read!"<<endl;
        return false;
    }
    cerr<<"\n";
    reader->Close();
    size_t valid_dim = 0;
    for (size_t i = 0; i < index_set.size(); i++) {
        if (index_set[i] == 1)
            valid_dim++;
    }
	cout << "data number  : " << dataNum << "\n";
	cout << "feat number  : " << featNum << "\n";
	cout << "dimension    : " << max_index << "\n";
	cout << "nonzero feat : " << valid_dim << "\n";
	cout << "class num    : " << map_class_sample_num.size() << "\n";
	if (map_class_sample_num.size() == 2){
		if (map_class_sample_num.find(1) != map_class_sample_num.end()) {
			int pos_num = map_class_sample_num[1];
			cout << "positive num	: " << pos_num << "\n";
		}
		if (map_class_sample_num.find(0) != map_class_sample_num.end()) {
			int neg_num = map_class_sample_num[0];
			cout << "negtive num	: " << neg_num << "\n";
		}
		else if (map_class_sample_num.find(-1) != map_class_sample_num.end()) {
			int neg_num = map_class_sample_num[-1];
			cout << "negtive num	: " << neg_num << "\n";
		}
	}
	if (max_index > 0){
		printf("data sparsity: %.2lf%%\n", 100 - valid_dim * 100.0 / max_index);
	}

	return true;
}

void InitParms(Params& param){

	string overview = "Sparse Online Learning Library - Dataset Analyzer";
	string syntax = "data_analysis -i input_file -st src_type";
	string example = "data_analysis -i input_file -st libsvm";
	param.Init(overview, syntax, example);

	//input & output
	param.add_option("", 1, 1, "input file", "-i", " ");
	param.add_option("", 1, 1, "input dataset type", "-st", " ");
}

int main(int argc, const char** args){

	//check memory leak in VC++
#if defined(_MSC_VER) && defined(_DEBUG)
	int tmpFlag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
	tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
	_CrtSetDbgFlag(tmpFlag);
#endif
	std::string ioInfo;
	IOInfo<float, char>::GetIOInfo(ioInfo);

	Params param;
	InitParms(param);
	if (param.Parse(argc, args) == false){
		return -1;
	}

	string filename = param.StringValue("-i");
	string src_type = param.StringValue("-st");
	//string filename = "/home/matthew/work/Data/aut/aut_train";
	DataReader<float, char> *reader = (DataReader<float, char>*)Registry::CreateObject(src_type, &filename);
	if (Analyze(reader) == false)
		cerr << "analyze dataset failed!" << endl;
	return 0;
}
