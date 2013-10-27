/*************************************************************************
  > File Name: DataSet_Converter.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Thu 26 Sep 2013 08:57:16 PM SGT
  > Functions: Convert the data set to our format
 ************************************************************************/

#pragma once
#include "data/DataPoint.h"
#include "data/DataReader.h"
#include "data/libsvmread.h"
#include "data/libsvm_binary.h"
#include "data/MNISTReader.h"


#include <string>
using namespace std;

bool Convert(DataReader<FeatType, LabelType> *reader, const string& cache_file) {
    if (reader == NULL)
        return false;

    libsvm_binary_<FeatType, LabelType> writer(cache_file);
    if (writer.OpenWriting() == false)
        return false;

    DataPoint<FeatType, LabelType> data;
    size_t dataNum = 0;
    if (reader->OpenReading() == true) {
        reader->Rewind();
        while(true) {
            if (reader->GetNextData(data) == true) {
                if (data.indexes.size() == 0)
                    continue;

                if (is_bc && data.label == 0)
                    data.label = -1;

                writer.WriteData(data);
                dataNum++;
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
    writer.Close();
    return true;
}
