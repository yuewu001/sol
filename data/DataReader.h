/*************************************************************************
	> File Name: DataReader.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 8/21/2013 Wednesday 4:48:28 PM
	> Functions: Interface for data reader
 ************************************************************************/

#pragma once

#include "DataPoint.h"
#include <vector>

namespace SOL
{
    template <typename FeatType, typename LabelType>
	class DataReader
	{
	public:
		virtual bool OpenReading() = 0;
		virtual bool GetNextData(DataPoint<FeatType, LabelType> &data) = 0;
		virtual void Rewind() = 0;
        virtual void Close() = 0;
	};

}
