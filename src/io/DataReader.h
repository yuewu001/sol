/*************************************************************************
  > File Name: DataReader.h
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: 8/21/2013 Wednesday 4:48:28 PM
  > Functions: Interface for data reader
 ************************************************************************/

#ifndef HEADER_DATAREADER
#define HEADER_DATAREADER

#include "DataPoint.h"
#include "../utils/reflector.h"

#include <string>

namespace BOC {

	#define IMPLEMENT_DATA_CLASS(className, name, descr) \
	template <typename FeatType, typename LabelType> \
	ClassInfo className<FeatType, LabelType>::classInfo(name, descr, className<FeatType, LabelType>::CreateObject); \
	\
	template <typename FeatType, typename LabelType> \
	void* className<FeatType, LabelType>::CreateObject(void* filename, void* param2, void* param3) \
	{ return new className<FeatType, LabelType>(*(std::string*)filename); }



	template <typename FeatType, typename LabelType>
	class DataReader : public Registry {
	protected:
		std::string fileName;

	public:
		DataReader(const std::string &fileName) {
			this->fileName = fileName;
		}

		virtual ~DataReader(){}
	public:
		/**
		* OpenReading: Open a dataset file and get it prepared to be read
		*
		* @Return: true if everything is ok
		*/
		virtual bool OpenReading() = 0;
		/**
		* GetNextData: for loading data sequentially
		*
		* @Param data: the variable to place the loaded data
		*
		* @Return: true if everything is ok
		*/
		virtual bool GetNextData(DataPoint<FeatType, LabelType> &data) = 0;
		/**
		* Rewind: Rewind the dataset to the beginning of the file 
		*/
		virtual void Rewind() = 0;

		/**
		* Close: Close the dataset when finished loading data
		*/
		virtual void Close() = 0;

		/**
		* Good : test the status of the data reader
		*
		* @Return: true if everything is ok
		*/
		virtual bool Good() = 0;
	};

}
#endif
