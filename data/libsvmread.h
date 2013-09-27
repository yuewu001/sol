/*************************************************************************
> File Name: libsvmread.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: 2013/8/18 星期日 20:25:28
> Functions: libsvm reader
************************************************************************/
#pragma once

#include "DataReader.h"

#include <stdio.h>
#include <vector>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <errno.h>
#include <limits>

using namespace std;

namespace SOL
{
	template <typename FeatType, typename LabelType>
    class LibSVMReader_: public DataReader<FeatType, LabelType>
	{ 
        private:
            string fileName;
            FILE* fp;

            char *line;
            int max_line_len;
            int min_index;


        public:
            LibSVMReader_(const string &fileName):fp(NULL)
        {
            this->max_line_len = 4096;
            this->fileName = fileName;
            this->min_index = numeric_limits<int>::max();

            line = (char *) malloc(max_line_len*sizeof(char));
        }
            ~LibSVMReader_()
            {
                this->Close();
                if (line != NULL)
                    free(line);
            }

            //////////////////online mode//////////////////
        public:
            virtual bool OpenReading()
            {
                this->Close();

                fp = fopen(fileName.c_str(), "r");
                if(fp == NULL)
                {
                    printf("can't open input file %s\n",fileName.c_str());
                    return false;
                }
                return true;
            }
            virtual void Rewind()
            {
                if(this->fp != NULL)
                    rewind(fp);
            }
            virtual void Close()
            {
                if (this->fp != NULL)
                    fclose(fp);
                fp = NULL;
            }

            virtual int GetMinIndex() const { return min_index; }

            virtual inline bool Good()
            {
                return fp != NULL;
            }

            virtual bool GetNextData(DataPoint<FeatType, LabelType> &data)
            {
                if (fp == NULL) 
                {
                    printf("no input file is opened for reading!\n");
                    return false;
                }

                if(readline(fp) == NULL)
                    return false;

                LabelType labelVal;
                char *idx, *val, *label;

                label = strtok(line," \t\n");
                if(label == NULL)
                {
                    printf("Empty line\n");
                    return false;
                }

                char *endptr;
                labelVal = (LabelType)strtod(label,&endptr);
                if(endptr == label || *endptr != '\0')
                    return false;

                int index;
                FeatType feat;
                // features
                while(1)
                {
                    idx = strtok(NULL,":");
                    val = strtok(NULL," \t");
                    if(val == NULL)
                        break;

                    index = (int)strtol(idx,&endptr,10) ; // precomputed kernel has <index> start from 0

                    if (index < 1)
                    {
                        printf("index should be no less than 1\n");
                    }
                    errno = 0;
                    feat =  strtod(val,&endptr);
                    if (endptr == val || errno != 0 || 
                            (*endptr != '\0' && !isspace(*endptr)))
                    {
                        printf("Wrong input format at line \n");
                        return false;
                    }
                    if (min_index > index)
                        min_index = index;

                    data.AddNewFeat(index,feat);
                }
                data.label = labelVal;
                return true;
            }

            ////////////Auxillary Functions//////////////
        private:
            char* readline(FILE *input)
            {
                int len;

                if(fgets(line,max_line_len,input) == NULL)
                    return NULL;

                while(strrchr(line,'\n') == NULL)
                {
                    max_line_len *= 2;
                    line = (char *) realloc(line, max_line_len);
                    len = (int) strlen(line);
                    if(fgets(line+len,max_line_len-len,input) == NULL)
                        break;
                }
                return line;
            }
    };

    //for special definition
    typedef LibSVMReader_<float, char> LibSVMReader;
}
