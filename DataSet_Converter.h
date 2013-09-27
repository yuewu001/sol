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
#include "Params.h"


#include <string>
using namespace std;

namespace SOL
{
    template <typename FeatType, typename LabelType> class DataSet_Converter_
    {
        public:
            static bool Convert(const Params &param)
            {
                string tmpFileName = "tmp~";
                libsvm_binary writer(tmpFileName);
                if (writer.OpenWriting() == false)
                    return false;

                DataReader<FeatType, LabelType> *reader = NULL;
                GetDataReader(param,reader);
                if (reader == NULL)
                    return false;

                bool is_bc = (param.data_type && DataSet_Type_BC) == 0;
                DataPoint<FeatType, LabelType> data;
                size_t dataNum = 0;
                if (reader->OpenReading() == true)
                {
                    reader->Rewind();
                    while(true)
                    {
                        data.erase();
                        if (reader->GetNextData(data) == true)
                        {
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
                else
                {
                    cerr<<"Can not open file to read!"<<endl;
                    return false;
                }
                int min_index = reader->GetMinIndex();
                reader->Close();
                writer.Close();

                return NormalizeData(min_index, tmpFileName, param.cache_fileName);
            }

        private:
            static bool NormalizeData(int min_index, const string& tmpFileName, const string &cache_file)
            {
                //check if the min_index is 1
                if (min_index == 1) //rename the temp file to 
                {
                    string cmd = "mv -f \"" + tmpFileName + "\" " + "\"" + cache_file + "\"";
                    int status = system(cmd.c_str());
                    return true;
                }
                else
                {
                    libsvm_binary inFile(tmpFileName);
                    libsvm_binary outFile(cache_file);
                    if (outFile.OpenWriting() == false)
                        return false;

                    if (inFile.OpenReading() == true)
                    {
                        int offset = min_index - 1;
                        DataPoint<FeatType, LabelType> data;
                        while(true)
                        {
                            data.erase();
                            if (inFile.GetNextData(data) == true)
                            {
                                int dim = data.indexes.size();
                                for (int i = 0; i < dim; i++)
                                    data.indexes[i] -= offset; 
                                data.max_index -= offset;
                                outFile.WriteData(data);
                            }
                            else
                                break;
                        }
                    }
                    else
                        return false;
                    return true;
                }
            }

            static void GetDataReader(const Params &param, DataReader<FeatType, LabelType> *&reader)
            {
                switch(param.data_type)
                {
                    case DataSet_LibSVM:
                        reader = new LibSVMReader_<FeatType, LabelType>(param.fileName);
                        break;
                    case DataSet_MNIST:
                        reader = new MNISTReader<FeatType,LabelType>(param.fileName,param.labelFile,param.digit_1,param.digit_2);
                        break;
                    default:
                        reader = NULL;
                        break;
                }
            }

            
    };

    typedef DataSet_Converter_<float,char> DataSet_Converter;
}

