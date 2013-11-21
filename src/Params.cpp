/*************************************************************************
  > File Name: Params.cpp
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Thu 26 Sep 2013 05:49:18 PM SGT
  > Functions: Class for Parsing parameters
 ************************************************************************/
#include "Params.h"
#include "common/util.h"
#include "common/init_param.h"

#include <iostream>
#include <cstdlib>

using namespace std;

namespace SOL {
    void ToUpperCase(string &str) {
        string dst_str;
        int len = str.length();
        for (int i = 0; i < len; i++)
            dst_str.push_back(toupper(str[i]));
        std::swap(str,dst_str);
    }

    void ToLowerCase(string &str) {
        string dst_str;
        int len = str.length();
        for (int i = 0; i < len; i++)
            dst_str.push_back(tolower(str[i]));
        std::swap(str,dst_str);
    }

    Params::Params(int argc, char** args) {
        this->Parse(argc, args);
    }

    void Params::Parse(int argc, char** args) {
        if (argc == 1){
            this->Help();
            exit(0);
        }

        this->Default();

        for (int i = 1; i < argc; i++) {
            if (strcmp(args[i],"-i") == 0 && i + 1 < argc)
                this->fileName = args[i + 1];
            else if (strcmp(args[i],"-c") == 0 && i + 1 < argc)
                this->cache_fileName = args[i + 1];
            else if (strcmp(args[i],"-t") ==  0 && i + 1 < argc)
                this->test_fileName = args[i+1];
            else if (strcmp(args[i],"-tc") == 0 && i + 1 < argc)
                this->test_cache_fileName = args[i + 1];
            else if (strcmp(args[i],"-passes") == 0 && i + 1 < argc)
                this->passNum = strtol(args[i+1], NULL, 10);
            else if (strcmp(args[i],"-l1") == 0 && i + 1 < argc)
                this->lambda = (float)strtod(args[i + 1], NULL);
            else if (strcmp(args[i],"-k") == 0 && i + 1 < argc)
                this->K = strtol(args[i + 1], NULL, 10);
            else if (strcmp(args[i],"-lbp") == 0)
                this->is_learn_best_param = true;
            else if (strcmp(args[i],"-eta") == 0 && i + 1 < argc)
                this->eta = (float)strtod(args[i+1], NULL);
            else if (strcmp(args[i], "-t0") == 0 && i + 1 < argc)
                this->initial_t = strtol(args[i+1], NULL, 10);
            else if (strcmp(args[i], "-power_t") == 0 && i + 1 < argc)
                this->power_t = (float)strtod(args[i+1], NULL);
            else if (strcmp(args[i], "-r") == 0 && i + 1 < argc)
                this->r = (float)strtod(args[i+1], NULL);
            else if (strcmp(args[i],"-delta") == 0 && i + 1 < argc)
                this->delta = (float)strtod(args[i + 1], NULL);
            else if (strcmp(args[i],"-opt") == 0 && i + 1 < argc) //opti method
                this->ParseOptiMethod(args[i+1]);
            else if (strcmp(args[i],"-dt") == 0 && i + 1 < argc) //data type
                this->ParseDataType(args[i + 1]);
            else if (strcmp(args[i],"-bs") == 0 && i + 1 < argc)
                this->buf_size = strtol(args[i+1], NULL, 10);
            else if(strcmp(args[i],"-loss") == 0 && i + 1 < argc)
                this->loss_type = this->GetLossType(args[i+1]);
            else if (strcmp(args[i],"-grou") == 0 && i + 1 < argc)
                this->gamma_rou = (float)strtod(args[i + 1], NULL);
            else if (strcmp(args[i],"--help") == 0)
                this->Help();
            else
                continue;
        }
        if (this->cache_fileName.size() == 0 && this->fileName.length() == 0){
            cerr<<"you must specify the training data"<<endl;
            exit(0);
        }
    }

    //default parameter settings
    void Params::Default() {
        this->data_type = init_data_type;
        this->opti_method = init_opti_method;
        this->loss_type = init_loss_type;
        this->passNum = 1;
        this->eta = -1;
        this->lambda = -1;
        this->delta = -1;
        this->K = -1;
        this->buf_size = -1;
        this->gamma_rou = init_gammarou;
        this->r = init_r;
        this->is_learn_best_param = init_is_learn_best_param;
        this->power_t = init_power_t;
        this->initial_t = init_initial_t;
    }

    void Params::ParseOptiMethod(char *str_method) {
        string str = str_method;
        ToUpperCase(str);
        const char* c_str = str.c_str();
        if (strcmp(c_str,"SGD") == 0)
            this->opti_method = Opti_SGD;
        else if (strcmp(c_str,"STG") == 0)
            this->opti_method = Opti_STG;
        else if (strcmp(c_str,"RDA") == 0)
            this->opti_method = Opti_RDA;
		else if (strcmp(c_str,"RDA_E") == 0)
			this->opti_method = Opti_RDA_E;
        else if (strcmp(c_str,"FOBOS") == 0)
            this->opti_method = Opti_FOBOS;
        else if (strcmp(c_str, "ADA-RDA") == 0)
            this->opti_method = Opti_Ada_RDA;
        else if (strcmp(c_str, "ADA-FOBOS") == 0)
            this->opti_method = Opti_Ada_FOBOS;
        else if (strcmp(c_str, "AROW") == 0)
            this->opti_method = Opti_DAROW;
        else if (strcmp(c_str, "SSAROW") == 0)
            this->opti_method = Opti_SSAROW;
        else if (strcmp(c_str, "ASAROW") == 0)
            this->opti_method = Opti_ASAROW;
        else {
            cerr<<"Unrecognized Optimization method!"<<endl;
            exit(0);
        }
    }

    void Params::ParseDataType(char *str_data_type) {
        string str = str_data_type;
        ToUpperCase(str);
        const char* c_str = str.c_str();
        if (strcmp(c_str,"LIBSVM") == 0) {
			this->data_type &= DataSet_Data_Type_Clear;
            this->data_type |= DataSet_LibSVM;
		}
        else {
            cout<<"Unrecognized Dataset Type!"<<endl;
            exit(0);
        }
    }

    enum_Loss_Type Params::GetLossType(char *str_type) {
        string str = str_type;
        ToLowerCase(str);
        const char* c_str = str.c_str();
        if (strcmp(c_str,"hinge") == 0)
            this->loss_type = Loss_Type_Hinge;
        else if (strcmp(c_str,"logit") == 0)
            this->loss_type = Loss_Type_Logit;
        else if (strcmp(c_str,"square") == 0)
            this->loss_type = Loss_Type_Square;
        else {
            cout<<"Unrecognized Loss Function Type!"<<endl;
            exit(0);
        }
        return  this->loss_type;

    }

    void Params::Help() {
        cout<<"SOL [options]\n";
        cout<<"Input Options: \n";
        cout<<"\t-i arg  :\t input file name\n";
        cout<<"\t-c arg  :\t cache file name\n";
        cout<<"\t-t arg  :\t test file name\n";
        cout<<"\t-tc arg :\t test cache file\n";
        cout<<"\t-dt arg :\t data type: LibSVM\n";
        cout<<"\t-bs arg :\t number of chunks for buffering\n";

        cout<<"Loss Functions: \n";
        cout<<"\t-loss arg:\t loss function type\n\t\t\t\tHinge|Logit|Square\n";

        cout<<"Algorithms and Parameters: \n";
        cout<<"\t-opt arg:\t optimization method:\n\t\t\t\tSGD|STG|RDA|RDA_E|FOBOS|Ada-RDA|Ada-FOBOS|AROW|SAROW\n";
        cout<<"\t-lbp    :\t learn best parameter\n";
        cout<<"\t-eta arg:\t learning rate\n";
        cout<<"\t-power_t arg:\tpower t of decaying learing rate\n";
        cout<<"\t-t0 arg:\t initial iteration number\n";
        cout<<"\t-l1 arg:\t value of l1 regularization\n";
        cout<<"\t-passes arg:\t number of passes\n\n";

        cout<<"\t-k arg:\t\t number of K in truncated gradient(STG)\n";
        cout<<"\t-grou arg:\t gamma times rou(RDA_E)\n";
        cout<<"\t-delta arg:\t value of delta for Adaptive algorithms(Ada-FOBOS, Ada-RDA)\n";

        cout<<"\n";
        exit(0);
    }
}
