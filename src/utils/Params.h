/*************************************************************************
> File Name: Params.h
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: Thu 26 Sep 2013 05:51:05 PM SGT
> Functions: Class for Parsing parameters
************************************************************************/

#ifndef HEADER_PARSER_PARAM
#define HEADER_PARSER_PARAM

#include <string>
#include <map>
#include <vector>

using std::string;
using std::map;

//using namespace ez;

namespace BOC {
	class Params {
	private:
        //separate ezOption definition, due to the error of ezOptionParser
		struct ezOption;
		struct ezOption *option;

		static const int max_param_num;
		float *float_storage;
		int* int_storage;
		bool* bool_storage;
		std::string* string_storage;
		int float_param_num;
		int int_param_num;
		int bool_param_num;
		int string_param_num;

		map<std::string, float*> flag2storage_float;
		map<std::string, int*> flag2storage_int;
		map<std::string, bool*> flag2storage_bool;
		map<std::string, std::string*> flag2storage_str;

		typedef map<std::string, float*>::iterator map_float_iter;
		typedef map<std::string, int*>::iterator map_int_iter;
		typedef map<std::string, bool*>::iterator map_bool_iter;
		typedef map<std::string, std::string*>::iterator map_str_iter;

	public:
		Params();
		~Params();

		bool Parse(int argc, const char** args);
		void Help();

	public:
		int IntValue(const std::string& param_name);
		float FloatValue(const std::string& param_name);
		bool BoolValue(const std::string& param_name);
		const std::string& StringValue(const std::string& param_name);

	private:
		//void Init(const std::map<std::string, std::vector<std::string> > &algoLossList);
		void Init();

		void add_option(float default_val, bool is_required, int expectArgs,
			const char* descr, const char* flag);
		void add_option(int default_val, bool is_required, int expectArgs,
			const char* descr, const char* flag);
		void add_option(bool default_val, bool is_required, int expectArgs,
			const char* descr, const char* flag);
		void add_option(const char* default_val, bool is_required, int expectArgs,
			const char* descr, const char* flag);
	};
}
#endif
