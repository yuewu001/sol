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
#include "common/init_param.h"
#include "Params.h"


#include <string>
using namespace std;

namespace SOL
{
    template <typename FeatType, typename LabelType> class DataSet_Converter
    {
        public:
            static bool Convert(const Params &param, bool is_test_file = false)
            {
                string cache_file;
                if (is_test_file == false)
                    cache_file = param.cache_fileName;
                else
                    cache_file = param.test_cache_fileName;

                libsvm_binary_<FeatType, LabelType> writer(cache_file);
                if (writer.OpenWriting() == false)
                    return false;

                DataReader<FeatType, LabelType> *reader = NULL;
                GetDataReader(param,reader,is_test_file);
                if (reader == NULL)
                    return false;

                bool is_bc = ((param.data_type && DataSet_Type_BC) != 0);
                DataPoint<FeatType, LabelType> data;
                size_t dataNum = 0;
                if (reader->OpenReading() == true)
                {
                    reader->Rewind();
                    while(true)
                    {
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

				if (min_index < 1)
				{
					cout<<"feature index should no less than 1!"<<endl;
					return false;
				}
                return true;
            }

        private:
            static bool NormalizeData(int min_index, const string &cache_file)
			{
#if WIN32
				string cmd = "ren \"" + init_tmp_file + "\" " + "\"" + cache_file + "\"";
#else
				string cmd = "mv -f \"" + init_tmp_file + "\" " + "\"" + cache_file + "\"";
#endif
				int status = system(cmd.c_str());
				if (status != 0)
				{
					cout<<"Convert data set failed!"<<endl;
					return false;
				}

				return true;
			}

            static void GetDataReader(const Params &param, DataReader<FeatType, LabelType> *&reader, bool is_test_file = false)
            {
				enum_DataSet_Type data_type = static_cast<enum_DataSet_Type>(param.data_type & DataSet_Work_Type_Clear);
                switch(data_type)
                {
                    case DataSet_LibSVM:
						if (is_test_file == false)
							reader = new LibSVMReader_<FeatType, LabelType>(param.fileName);
						else
							reader = new LibSVMReader_<FeatType, LabelType>(param.test_fileName);
                        break;
                    case DataSet_MNIST:
						{
						if (is_test_file == false)
							reader = new MNISTReader<FeatType,LabelType>(param.fileName,param.labelFile,param.digit_1,param.digit_2);
						else
							reader = new MNISTReader<FeatType,LabelType>(param.test_fileName,param.test_label_fileName,param.digit_1,param.digit_2);

                        break;
						}
                    default:
                        reader = NULL;
                        break;
                }
            }

            
    };
}

