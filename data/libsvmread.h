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
#include <string>
#include <stdlib.h>
#include <ctype.h>
#include <errno.h>


using namespace std;

namespace SOL
{
	template <typename FeatType, typename LabelType>
	class LibSVMReader_: public DataReader<FeatType, LabelType>
	{ 
		string fileName;
        FILE* fp;

        char *line;
		int max_line_len;


	public:
		LibSVMReader_(const string &fileName):fp(NULL)
        {
            this->max_line_len = 4096;
            this->fileName = fileName;

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

		virtual bool GetNextData(DataPoint<FeatType, LabelType> &data)
        {
            if (fp == NULL) 
            {
                printf("no input file is opened for reading!\n");
                return false;
            }
            char *idx, *val, *label;
            if(readline(fp) == NULL)
                return false;

            label = strtok(line," \t\n");
            if(label == NULL)
            {
                printf("Empty line at line \n");
                return false;
            }

            char *endptr;
            LabelType labelVal = (LabelType)strtod(label,&endptr);
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
                    return false;
                }
                errno = 0;
                feat =  strtod(val,&endptr);
                if (endptr == val || errno != 0 || 
                        (*endptr != '\0' && !isspace(*endptr)))
                {
                    printf("Wrong input format at line \n");
                    return false;
                }
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
