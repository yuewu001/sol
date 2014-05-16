/*************************************************************************
	> File Name: DataHandler.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 2/17/2014 5:03:34 PM
	> Functions: interface for data reader and data writer
	************************************************************************/

#ifndef HEADER_DATA_HANDLER
#define HEADER_DATA_HANDLER

#include "DataReader.h"

namespace BOC {
	template <typename FeatType, typename LabelType>
	class DataHandler : public DataReader<FeatType, LabelType> {
	public:
		DataHandler(const std::string &fileName) : DataReader<FeatType, LabelType>(fileName){}
		virtual ~DataHandler(){}
	public:
		/**
		 * OpenWriting: Open a dataset file and get it prepared to be written
		 *
		 * @Return: true if everything is ok
		 */
		virtual bool OpenWriting() = 0;

		/**
		 *  WriteData : write a new data into the file
		 *
		 * @Param:  data
		 *
		 * @Returns: true if succeed
		 */
		virtual bool WriteData(DataPoint<FeatType, LabelType> &data) = 0;

		/**
		*  SetExtraInfo: set extra information for writer
		*  this function is used for some special data formats like csv,arff
		*/
		virtual bool SetExtraInfo(const char* info_stream) { return true; };
	};

}

#endif
