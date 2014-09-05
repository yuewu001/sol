/*************************************************************************
  > File Name: dtcleaner.cpp
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Wed 12 Feb 2014 08:09:38 PM
  > Descriptions: remove features never appear
 ************************************************************************/
#include "../io/DataPoint.h"
#include "../io/DataReader.h"
#include "../io/libsvm_io.h"
#include "../utils/util.h"

#include <string>
using namespace std;
using namespace BOC;

bool Detect(const string& filename, s_array<char> &index_set) {
	libsvm_io<float, char> reader(filename);

	size_t max_show_count = 100000;
	size_t show_count = 1000;
    size_t dataNum = 0;
    IndexType max_index = 0;
    index_set.resize(0);
    DataPoint<float,char> data;
    if (reader.OpenReading() == true) {
        reader.Rewind();
        while(true) {
            if (reader.GetNextData(data) == true) {
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
	size_t valid_dim = 0;
	for (size_t i = 0; i < index_set.size(); i++) {
		if (index_set[i] == 1)
			valid_dim++;
	}
	cout<<"data number  : "<<dataNum<<"\n";
	cout<<"valid dim    : "<<max_index<<"\n";
	cout<<"nonzero feat : "<<valid_dim<<"\n";
	if (max_index > 0){
		printf("data sparsity: %.2lf%%\n",100 - valid_dim * 100.0 / max_index);
	}

	return true;
}
void Convert(const string& in_filename, const string& out_filename, const s_array<IndexType> &index_set){
    libsvm_io<float, char> reader(in_filename);

	cout<<"remove useless features not appeared "<<endl;
	if (reader.OpenReading() == false){
		cerr<<"open "<<in_filename<<" failed!"<<endl;
		return;
	}
	string tmp_filename = out_filename + ".writing";

	libsvm_io<float,char> writer(tmp_filename);
	if(writer.OpenWriting() == false){
		cerr<<"open output file" <<tmp_filename<<" failed!"<<endl;
		return;
	}
	DataPoint<float, char> data;
	size_t dataNum = 0;
	size_t show_step = 1; //show information every show_step
	size_t show_count = 2;
	while(reader.GetNextData(data) == true){
		dataNum++;

        size_t featNum = data.indexes.size();
        for (size_t i = 0; i < featNum; i++){
			if (data.indexes[i] < index_set.size() && index_set[data.indexes[i] - 1] != 0)
				data.indexes[i] = index_set[data.indexes[i] - 1];
        }

		if (writer.WriteData(data) == false){
			break;
		}

		if (show_count < dataNum){
			printf("%lu samples processed\r",dataNum);
			show_count = ((size_t)(1) << ++show_step);
		}
	}
	writer.Close();
	if (reader.Good() == true && 
		rename_file(tmp_filename,out_filename) == true)
		printf("%lu samples processed\n",dataNum);
	reader.Close();
}

int main(int argc, char** args){ 
	if (argc < 3){
		cout<<"Usage: dtcleaner in_file out_file"<<endl;
		return 0;
	}
	
	//check memory leak in VC++
#if defined(_MSC_VER) && defined(_DEBUG)
	int tmpFlag = _CrtSetDbgFlag( _CRTDBG_REPORT_FLAG );
	tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
	_CrtSetDbgFlag( tmpFlag );
#endif
	string filename = args[1];
	string out_filename = args[2];
	//string filename = "/home/matthew/work/Data/aut/aut_train";
    s_array<char> index_set;
	if (Detect(filename,index_set) == false)
		cerr<<"analyze dataset failed!"<<endl;
    s_array<IndexType> new_index;
    size_t featDim = index_set.size();
    new_index.resize(featDim);
    new_index.zeros();
    IndexType index = 1;
    for (size_t i = 0; i < featDim; i++){
        if (index_set[i] == 1){
            new_index[i] = index++;
        }
    }
	Convert(filename, out_filename, new_index);
	return 0;
}
