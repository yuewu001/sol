/*************************************************************************
  > File Name: data_analysis.cpp
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Thu 24 Oct 2013 08:09:38 PM
  > Descriptions: analyse the sparsity of data
 ************************************************************************/
#include "DataPoint.h"
#include "DataReader.h"
#include "libsvmread.h"
#include "MNISTReader.h"

#include <string>
#include <set>
using namespace std;
using namespace SOL;

template <typename FeatType, typename LabelType>
bool Analyze(DataReader<FeatType, LabelType> *reader) {
    if (reader == NULL){
        cerr<<"data reader is emptyp!"<<endl;
        return false;
    }

    size_t dataNum = 0;
    size_t featNum = 0;
    IndexType max_index = 0;
    set<IndexType> index_set;
    DataPoint<FeatType, LabelType> data;
    if (reader->OpenReading() == true) {
        reader->Rewind();
        while(true) {
            if (reader->GetNextData(data) == true) {
                if (data.indexes.size() == 0)
                    continue;
                for (size_t i = 0; i < data.indexes.size(); i++){
                    index_set.insert(data.indexes[i]);
                }
                if (max_index < data.dim()){
                    max_index = data.dim();
                }

                dataNum++;
                featNum += data.indexes.size();
                /*
                if (dataNum % 1000 == 0){
                    cout<<"data number  : "<<dataNum<<"    ";
                    cout<<"valid dim    : "<<max_index<<"\r";
                }
                */
            }
            else
                break;
        }
    }
    else {
        cerr<<"Can not open file to read!"<<endl;
        return false;
    }
    reader->Close();
    cout<<"data number  : "<<dataNum<<"\n";
    cout<<"feat number  : "<<featNum<<"\n";
    cout<<"valid dim    : "<<max_index<<"\n";
    cout<<"nonzero feat : "<<index_set.size()<<"\n";
    if (max_index > 0){
        printf("data sparsity: %.2lf%%\n",100 - index_set.size() * 100.0 / max_index);
    }

    return true;
}

int main(int argc, char** args){ 
    if (argc != 2){
        cout<<"Usage: data_analysis data_file"<<endl;
        return 0;
    }
//check memory leak in VC++
#if defined(_MSC_VER) && defined(_DEBUG)
	int tmpFlag = _CrtSetDbgFlag( _CRTDBG_REPORT_FLAG );
	tmpFlag |= _CRTDBG_LEAK_CHECK_DF;
	_CrtSetDbgFlag( tmpFlag );
#endif
    string filename = args[1];
    //string filename = "/home/matthew/work/Data/aut/aut_train";
    LibSVMReader reader(filename);
    if (Analyze(&reader) == false)
        cerr<<"analyze dataset failed!"<<endl;
    return 0;
}
