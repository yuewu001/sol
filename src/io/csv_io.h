/*************************************************************************
	> File Name: csv_io.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2/17/2014 4:40:34 PM
	> Functions: csv file io handler
 ************************************************************************/

#ifndef HEADER_CSV_IO_HANDLER
#define HEADER_CSV_IO_HANDLER
#include "DataHandler.h"
#include "basic_io.h"
#include "parser.h"

#include <stdio.h>
#include <vector>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <cmath>
#include <sstream>

using namespace std;

namespace SOL {
    template <typename FeatType, typename LabelType>
        class csv_io_: public DataHandler<FeatType, LabelType> { 
            private:
                string fileName;
                basic_io io_hander;
				FILE* writer_handler;

                char *line;
                size_t max_line_len;

				bool is_good;

                IndexType featDim = 0;

            public:
                csv_io_(const string &fileName) {
                    this->max_line_len = 4096;
                    this->fileName = fileName;
                    line = (char *) malloc(max_line_len*sizeof(char));
					this->is_good = true;
					this->writer_handler = NULL;
                }
                virtual ~csv_io_() {
                    this->Close();
                    if (line != NULL)
                        free(line);
                }

                //////////////////online mode//////////////////
            public:
                virtual bool OpenReading() {
                    this->Close();
					this->is_good = io_hander.open_file(this->fileName.c_str(), "rb");
					if (this->is_good)
						return this->LoadFeatDim();
					return false;
				}
				bool OpenWriting() {
					this->Close();
					this->writer_handler = fopen(this->fileName.c_str(), "wb");
					this->is_good = this->writer_handler != NULL ? true : false;
					return this->is_good;
				}

				virtual void Rewind() {
					io_hander.rewind();
					io_hander.read_line(line, max_line_len); //read the first line for csv
				}
				virtual void Close() {
					io_hander.close_file();
					if (this->writer_handler != NULL) {
						fclose(this->writer_handler);
						this->writer_handler = NULL;
					}
				}

				virtual inline bool Good() {
					return this->is_good == true && io_hander.good() == 0 ? true : false;
				}

				virtual bool GetNextData(DataPoint<FeatType, LabelType> &data) {
					if (io_hander.read_line(line, max_line_len) == NULL)
						return false;

					LabelType labelVal;
					char* p = line, *endptr = NULL;
					if (*p == '\0')
						return false;
					labelVal = (LabelType)parseInt_CSV(p, &endptr);
					if (endptr == p) {
						fprintf(stderr, "parse label failed.\n");
						this->is_good = false;
						return false;
					}

					data.erase();
					IndexType index(0);
					FeatType feat;
					// features
					while (1) {
						p = strip_line(endptr);
						while (*p == ','){
							index++;
							p++;
							p = strip_line(p);
						}
						if (*p == '\0')
							break;
						feat = parseFloat_CSV(p, &endptr);
						//feat =(float)(strtod(val,&endptr));
						if (endptr == p) {
							fprintf(stderr, "parse feature value failed!\n");
							this->is_good = false;
							return false;
						}

						data.AddNewFeat(index, feat);
					}
					data.label = labelVal;

					return true;
				}

				bool WriteData(DataPoint<FeatType, LabelType> &data) {
					size_t featNum = data.indexes.size();
					fprintf(writer_handler, "%d", data.label);
					IndexType i = 0, j = 1;
					for (; i < featNum, j <= featDim; j++){
						fprintf(writer_handler, ",");
						if (data.indexes[i] == j)
							fprintf(writer_handler, "%g", data.features[i++]);
					}
					for (; j <= featDim; j++)
						fprintf(writer_handler, ",");

					fprintf(writer_handler, "\n");
					return true;
				}
				/**
				*  SetExtraInfo: set extra information for writer
				*  this function is used for some special data formats like csv,arff
				*	Here, info_stream is the data dimension
				*/
				virtual bool SetExtraInfo(const char* info_stream) {
					int err_code = ferror(this->writer_handler);
					if (err_code != 0){
						fprintf(stderr, "something error occured on csv write. error code %d\n", err_code);
						return false;
					}
					this->featDim = *((IndexType*)(info_stream));
					std::ostringstream oss;
					oss << "class";
					for (IndexType i = 1; i <= this->featDim; i++){
						oss << ",v" << i;
					}
					oss << "\n";
					fwrite(oss.str().c_str(), 1, oss.str().length(), this->writer_handler);
					return true;
				}

		protected:
			bool LoadFeatDim(){
				if (io_hander.read_line(line, max_line_len) == NULL)
					return false;
				char* p = line;
				this->featDim = 0;
				while (*p != '\0'){
					if (*p++ == ',')
						this->featDim++;
				}
			}
		};

		//for special definition
		typedef csv_io_<float, char> csv_io;
}



#endif
