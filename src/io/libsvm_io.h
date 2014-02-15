/*************************************************************************
  > File Name: libsvmread.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: 2013/8/18 Sunday 20:25:28
  > Functions: libsvm io_hander
 ************************************************************************/
#ifndef HEADER_LIBSVM_READ
#define HEADER_LIBSVM_READ

#if _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "DataReader.h"
#include "basic_io.h"
#include "parser.h"

#include <stdio.h>
#include <vector>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <cmath>

using namespace std;

namespace SOL {
    template <typename FeatType, typename LabelType>
        class libsvm_io_: public DataReader<FeatType, LabelType> { 
            private:
                string fileName;
                basic_io io_hander;
				FILE* writer_handler;

                char *line;
                size_t max_line_len;

				bool is_good;

            public:
                libsvm_io_(const string &fileName) {
                    this->max_line_len = 4096;
                    this->fileName = fileName;
                    line = (char *) malloc(max_line_len*sizeof(char));
					this->is_good = true;
					this->writer_handler = NULL;
                }
                virtual ~libsvm_io_() {
                    this->Close();
                    if (line != NULL)
                        free(line);
                }

                //////////////////online mode//////////////////
            public:
                virtual bool OpenReading() {
                    this->Close();
					this->is_good = io_hander.open_file(this->fileName.c_str(), "rb");
					return this->is_good;
                }
				bool OpenWriting() {
					this->Close();
#if _WIN32
					errno_t ret = fopen_s(&this->writer_handler, this->fileName.c_str(),"wb");
					if (ret != 0){
						printf("error %d: can't open file %s\n", ret, this->fileName.c_str());
						this->is_good = false;
					}
					else
						this->is_good = true;
#else
					this->writer_handler = fopen(this->fileName.c_str(), "wb");
					this->is_good = this->writer_handler != NULL ? true : false;
#endif
					return this->is_good;
				}

				virtual void Rewind() {
					io_hander.rewind();
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
					labelVal = (LabelType)parseInt(p, &endptr);
					if (endptr == p) {
						fprintf(stderr, "parse label failed.\n");
						this->is_good = false;
						return false;
					}

					data.erase();
					IndexType index;
					FeatType feat;
					// features
					while (1) {
						p = strip_line(endptr);
						if (*p == '\0')
							break;
						index = (IndexType)(parseUint(p, &endptr));
						if (endptr == p) { //parse index failed
							fprintf(stderr, "parse index value failed!\n%s", p);
							this->is_good = false;
							return false;
						}

						p = endptr;
						feat = parseFloat(p, &endptr);
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
					for (IndexType i = 0; i < featNum; i++){
						fprintf(writer_handler, " %d:%f", data.indexes[i], data.features[i]);
					}
					fprintf(writer_handler, "\n");
					return true;
				}
		};

		//for special definition
		typedef libsvm_io_<float, char> libsvm_io;
}

#endif
