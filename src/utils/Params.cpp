/*************************************************************************
> File Name: Params.cpp
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: Thu 26 Sep 2013 05:49:18 PM SGT
> Functions: Class for Parsing parameters
************************************************************************/
#include "ezOptionParser.hpp"
#include "Params.h"
#include "util.h"

#include <stdlib.h>
#include <iostream>
#include <stdexcept>

using namespace std;
using namespace ez;

namespace BOC {
	const int Params::max_param_num = 256;

	struct Params::ezOption{
		ez::ezOptionParser opt;
		ez::ezOptionValidator* vfloat;
		ez::ezOptionValidator* vint;
		ez::ezOptionValidator* vbool;

		ezOption(){
			this->vfloat = new ezOptionValidator("f", "ge", "0");
			this->vint = new ezOptionValidator("u4", "ge", "0");
			this->vbool = new ezOptionValidator("t", "in", "true,false", false);
		}
		~ezOption(){
			if (this->vfloat != NULL) delete this->vfloat;
			this->vfloat = NULL;
			if (this->vint != NULL) delete this->vint;
			this->vint = NULL;
			if (this->vbool != NULL) delete this->vbool;
			this->vbool = NULL;
		}
	};

	Params::Params() {
		this->option = new ezOption;

		this->float_storage = new float[max_param_num];
		this->int_storage = new int[max_param_num];
		this->bool_storage = new bool[max_param_num];
		this->string_storage = new string[max_param_num];

		this->float_param_num = 0;
		this->int_param_num = 0;
		this->bool_param_num = 0;
		this->string_param_num = 0;
	}

	Params::~Params(){
		if (this->float_storage != NULL) delete[]this->float_storage;
		if (this->int_storage != NULL) delete[]this->int_storage;
		if (this->bool_storage != NULL) delete[]this->bool_storage;
		if (this->string_storage != NULL) delete[]this->string_storage;
		if (this->option != NULL) {
			delete this->option;
			this->option = NULL;
		}
	}

	//void Params::Init(const std::map<std::string, std::vector<std::string> > &algoLossList){
	void Params::Init(const string& overview, const string &syntax, const string& example){
		//initialize params
		this->option->opt.overview = overview;
		option->opt.syntax = syntax;
		option->opt.example = example;

		option->opt.add("", " ", 0, 0, ',', "help message", "-h", "--help");
	}

	void Params::add_option(float default_val, bool is_required, int expectArgs,
		const char* descr, const char* flag, const char* category){
		this->option->opt.add("", category, is_required, expectArgs, 0, descr, flag, this->option->vfloat);
		if (this->float_param_num < max_param_num){
			this->float_storage[this->float_param_num] = default_val;
			this->flag2storage_float[flag] = this->float_storage + this->float_param_num;
			this->float_param_num++;
		}
		else{
			throw runtime_error("number of parameters exceed the upper limit!");
		}
	}

	void Params::add_option(int default_val, bool is_required, int expectArgs,
		const char* descr, const char* flag, const char* category){
		this->option->opt.add("", category, is_required, expectArgs, 0, descr, flag, this->option->vint);
		if (this->int_param_num < max_param_num){
			this->int_storage[this->int_param_num] = default_val;
			this->flag2storage_int[flag] = this->int_storage + this->int_param_num;
			this->int_param_num++;
		}
		else{
			throw runtime_error("number of parameters exceed the upper limit!");
		}
	}

	void Params::add_option(bool default_val, bool is_required, int expectArgs,
		const char* descr, const char* flag,const char* category){
		this->option->opt.add("", category, is_required, expectArgs, 0, descr, flag, this->option->vbool);
		if (this->bool_param_num < max_param_num){
			this->bool_storage[this->bool_param_num] = default_val;
			this->flag2storage_bool[flag] = this->bool_storage + this->bool_param_num;
			this->bool_param_num++;
		}
		else{
			throw runtime_error("number of parameters exceed the upper limit!");
		}
	}

	void Params::add_option(const char* default_val, bool is_required, int expectArgs,
		const char* descr, const char* flag, const char* category){
		this->option->opt.add("", category, is_required, expectArgs, 0, descr, flag);
		if (this->string_param_num < max_param_num){
			this->string_storage[this->string_param_num] = default_val;
			this->flag2storage_str[flag] = this->string_storage + this->string_param_num;
			this->string_param_num++;
		}
		else{
			throw runtime_error("number of parameters exceed the upper limit!");
		}
	}

	bool Params::Parse(int argc, const char** args) {
		option->opt.parse(argc, args);
		if (option->opt.isSet("-h") || argc == 1){
			this->Help();
			return false;
		}
		vector<string> badOptions;
		if (!option->opt.gotRequired(badOptions)){
			for (size_t i = 0; i < badOptions.size(); i++)
				cerr << "ERROR: Missing required option " << badOptions[i] << ".\n";
			this->Help();
			return false;
		}
		if (!option->opt.gotExpected(badOptions)){
			for (size_t i = 0; i < badOptions.size(); i++)
				cerr << "ERROR: Got unexpected number of arguments for option " << badOptions[i] << ".\n";
			this->Help();
			return false;
		}
		vector<string> badArgs;
		if (!option->opt.gotValid(badOptions, badArgs)) {
			for (size_t i = 0; i < badOptions.size(); ++i)
				std::cerr << "ERROR: Got invalid argument \"" << badArgs[i] << "\" for option " << badOptions[i] << ".\n";
			return false;
		}
		for (map_float_iter iter = this->flag2storage_float.begin();
			iter != this->flag2storage_float.end(); iter++){
			if (option->opt.isSet(iter->first.c_str()))
				option->opt.get(iter->first.c_str())->getFloat(*(iter->second));
		}

		for (map_int_iter iter = this->flag2storage_int.begin();
			iter != this->flag2storage_int.end(); iter++){
			if (option->opt.isSet(iter->first.c_str()))
				option->opt.get(iter->first.c_str())->getInt(*(iter->second));
		}
		for (map_bool_iter iter = this->flag2storage_bool.begin();
			iter != this->flag2storage_bool.end(); iter++){
			if (option->opt.isSet(iter->first.c_str())){
				if (option->opt.get(iter->first.c_str())->expectArgs == 0)
					*(iter->second) = true;
				else{
					string out;
					option->opt.get(iter->first.c_str())->getString(out);
					ToLowerCase(out);
					if (out == "true")
						*(iter->second) = true;
					else
						*(iter->second) = false;
				}
			}
		}
		for (map_str_iter iter = this->flag2storage_str.begin();
			iter != this->flag2storage_str.end(); iter++){
			if (option->opt.isSet(iter->first.c_str()))
				option->opt.get(iter->first.c_str())->getString(*(iter->second));
		}

		return true;
	}

	int Params::IntValue(const std::string& param_name) {
		map_int_iter iter = this->flag2storage_int.find(param_name);
		if (iter != this->flag2storage_int.end()){
			return *(iter->second);
		}
		else{
			string errMsg = param_name + " is not found in parameter list!";
			throw invalid_argument(errMsg.c_str());
		}
		return 0;
	}

	float Params::FloatValue(const std::string& param_name) {
		map_float_iter iter = this->flag2storage_float.find(param_name);
		if (iter != this->flag2storage_float.end()){
			return *(iter->second);
		}
		else{
			string errMsg = param_name + " is not found in parameter list!";
			throw invalid_argument(errMsg.c_str());
		}
		return 0;
	}

	bool Params::BoolValue(const std::string& param_name) {
		map_bool_iter iter = this->flag2storage_bool.find(param_name);
		if (iter != this->flag2storage_bool.end()){
			return *(iter->second);
		}
		else{
			string errMsg = param_name + " is not found in parameter list!";
			throw invalid_argument(errMsg.c_str());
		}
		return 0;
	}

	const std::string& Params::StringValue(const std::string& param_name) {
		map_str_iter iter = this->flag2storage_str.find(param_name);
		if (iter != this->flag2storage_str.end()){
			return *(iter->second);
		}
		else{
			string errMsg = param_name + " is not found in parameter list!";
			throw invalid_argument(errMsg.c_str());
		}
	}

	void Params::Help() {
		string usage;
		option->opt.getUsageByCategory(usage);
		cout << usage << endl;
	}


}
