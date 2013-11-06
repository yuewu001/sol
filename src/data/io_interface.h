/*************************************************************************
  > File Name: io_interface.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Wed 06 Nov 2013 03:26:20 PM
  > Descriptions: interface definition for io
 ************************************************************************/
#ifndef HEADER_IO_INTERFACE_
#define HEADER_IO_INTERFACE_

namespace SOL{
    class io_interface{
        public:
            virtual ~io_handler(){
                this->close_file();
            }

        public:
            virtual bool open_file(const char* filename, const char* mode) = 0;
            // bind_stdin: bind the input to stdin
            virtual bool open_stdin() = 0;
            // bind_stdin: bind the output to stdout
            virtual bool open_stdout() = 0;

            virtual void close_file() = 0;
            virtual void rewind() = 0;
            /**
             * good : test if the io is good
             *
             * @Return: zero if correct, else zero code
             */
            virtual int good() = 0;

        public:
            /**
             * read_data : read the data from file
             *
             * @Param dst: container to place the read data
             * @Param length: length of data of read in bytes
             *
             * @Return: size of data read in bytes
             */
            virtual size_t read_data(char* dst, size_t length) = 0;

            /**
             * read_line : read a line from disk
             *
             * @Param dst: container to place the read data
             * @Param dst_len: length of dst
             *
             * @Return: pointer to the read line, null if failed
             */
            virtual char* read_line(char* &dst, size_t &dst_len) = 0;
            
            /**
             * write_data : write content to disk
             *
             * @Param src: source of the data
             * @Param length: length to write the data
             *
             * @Return: size of data written to disk in bytes
             */
            virtual size_t write_data(char* src, size_t length) = 0;
    };
}

#endif
