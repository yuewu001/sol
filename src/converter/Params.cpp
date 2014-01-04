/*************************************************************************
> File Name: Params.cpp
> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
> Created Time: Thu 26 Sep 2013 05:49:18 PM SGT
> Functions: Class for Parsing parameters
************************************************************************/
#include "Params.h"
#include "../utils/util.h"
#include "../utils/init_param.h"

#include <iostream>
#include <cstdlib>

using namespace std;
using namespace ez;

namespace SOL {
	Params::Params() {
		this->vfloat = new ezOptionValidator("f");
		this->vint = new ezOptionValidator("u4");
		this->vbool = new ezOptionValidator("t","in","true,false",false);

		this->Init();
	}

	Params::~Params(){
	}
	void Params::Init(){
		//initialize params
		opt.overview = "Dataset Converter of Sparse Online Learning Library";
		opt.syntax	= "SOL -i input_file -o output_file [options]" ;
		opt.example = "SOL -i input_file -o output_file";

		opt.add("",0,0,',',"help message","-h","--help");

		this->add_option("",true,1,"input file","-i", &this->in_fileName);
		this->add_option("",true,1,"output file name","-o",&this->out_fileName);

		this->add_option(init_data_type,0,1,"dataset type format","-dt",&this->str_data_type);

		this->add_option(false,0,0,"cache the input file","-c",&this->is_cache);
		this->add_option(false,0,0,"de-cache the input file","-d",&this->is_de_cache);
	}

	void Params::add_option(float default_val, bool is_required, int expectArgs, 
		const char* descr, const char* flag, float *storage){
			*storage = default_val;
			this->opt.add("",is_required,expectArgs,0,descr,flag,this->vfloat);
			this->flag2storage_float[flag] = storage;
	}
	void Params::add_option(int default_val, bool is_required, int expectArgs, 
		const char* descr, const char* flag, int *storage){
			*storage = default_val;
			this->opt.add("",is_required,expectArgs,0,descr,flag,this->vint);
			this->flag2storage_int[flag] = storage;
	}
	void Params::add_option(bool default_val, bool is_required, int expectArgs, 
		const char* descr, const char* flag, bool *storage){
			*storage = default_val;
			this->opt.add("",is_required,expectArgs,0,descr,flag, this->vbool);
			this->flag2storage_bool[flag] = storage;
	}

	void Params::add_option(const char* default_val, bool is_required, int expectArgs, 
		const char* descr, const char* flag, string *storage){
			*storage = default_val;
			this->opt.add("",is_required,expectArgs,0,descr,flag);
			this->flag2storage_str[flag] = storage;
	}

	bool Params::Parse(int argc, const char** args) {
		if (opt.isSet("-h") || argc == 1){
			this->Help();
			return false;
		}
		opt.parse(argc, args);
		vector<string> badOptions;
		if (!opt.gotRequired(badOptions)){
			for (size_t i = 0; i < badOptions.size(); i++)
				cerr<<"ERROR: Missing required option "<<badOptions[i]<<".\n\n";
			this->Help();
			return false;
		}
		if (!opt.gotExpected(badOptions)){
			for (size_t i = 0; i < badOptions.size(); i++)
				cerr<<"ERROR: Got unexpected number of arguments for option "<<badOptions[i]<<".\n\n";
			this->Help();
			return false;
		}
		for (map_float_iter iter = this->flag2storage_float.begin();
			iter != this->flag2storage_float.end(); iter++){
				if (opt.isSet(iter->first.c_str()))
					opt.get(iter->first.c_str())->getFloat(*(iter->second));
		}

		for (map_int_iter iter = this->flag2storage_int.begin();
			iter != this->flag2storage_int.end(); iter++){
				if (opt.isSet(iter->first.c_str()))
					opt.get(iter->first.c_str())->getInt(*(iter->second));
		}
		for (map_bool_iter iter = this->flag2storage_bool.begin();
			iter != this->flag2storage_bool.end(); iter++){
            if (opt.isSet(iter->first.c_str())){
                if (opt.get(iter->first.c_str())->expectArgs == 0)
                    *(iter->second) = true;
                else{
                    string out;
                    opt.get(iter->first.c_str())->getString(out);
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
				if (opt.isSet(iter->first.c_str()))
					opt.get(iter->first.c_str())->getString(*(iter->second));
		}

		return true;
	}

	void Params::Help() {
		string usage;
		opt.getUsage(usage);
		cout<<usage<<endl;
	}
}
