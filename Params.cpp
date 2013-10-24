/*************************************************************************
  > File Name: Params.cpp
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Thu 26 Sep 2013 05:49:18 PM SGT
  > Functions: Class for Parsing parameters
 ************************************************************************/
#include "Params.h"
#include "common/md5.h"
#include "common/util.h"
#include "common/init_param.h"

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
                this->passNum = atoi(args[i+1]);
            else if (strcmp(args[i],"-l1") == 0 && i + 1 < argc)
                this->lambda = strtod(args[i + 1],NULL);
            else if (strcmp(args[i],"-k") == 0 && i + 1 < argc)
                this->K = atoi(args[i + 1]);
            else if (strcmp(args[i],"-eta") == 0 && i + 1 < argc)
                this->eta = strtod(args[i+1],NULL);
            else if (strcmp(args[i],"-theta") == 0 && i + 1 <argc)
                this->theta = strtod(args[i+1],NULL);
            else if (strcmp(args[i], "-r") == 0 && i + 1 < argc)
                this->r = strtod(args[i+1], NULL);
            else if (strcmp(args[i],"-delta") == 0 && i + 1 < argc)
                this->delta = strtod(args[i + 1],NULL);
            else if (strcmp(args[i],"-opt") == 0 && i + 1 < argc) //opti method
                this->ParseOptiMethod(args[i+1]);
            else if (strcmp(args[i],"-dt") == 0 && i + 1 < argc) //data type
                this->ParseDataType(args[i + 1]);
            else if (strcmp(args[i],"-bs") == 0 && i + 1 < argc)
                this->buf_size = atoi(args[i+1]);
            else if(strcmp(args[i],"-loss") == 0 && i + 1 < argc)
                this->loss_type = this->GetLossType(args[i+1]);
            else if (strcmp(args[i],"-gamma") == 0 && i + 1 < argc)
                this->gamma = strtod(args[i+1],NULL);
            else if (strcmp(args[i],"-rou") == 0 && i + 1 < argc)
                this->rou = strtod(args[i + 1],NULL);
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
        this->theta = -1;
        this->delta = -1;
        this->K = -1;
        this->buf_size = -1;
        this->gamma = -1;
        this->rou = -1;
        this->r = init_r;
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
            this->opti_method = Opti_AROW;
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
        cout<<"SOL -i input_data [-c cache_file] [option | value]\n";
        cout<<"Options: \n";
        cout<<"-c:\t cache file name\n";
        cout<<"-t:\t test file name\n";
        cout<<"-tc:\t test cache file\n";
        cout<<"-dt:\t data type: LibSVM\n";
        cout<<"-bs:\t number of chunks for buffering\n";
        cout<<"-eta:\t learning rate\n";
        cout<<"-delta:\t value of delta for Adaptive algorithms\n";
        cout<<"-gamma\t gamma\n";
        cout<<"-k:\t number of K in truncated gradient\n";
        cout<<"-l1:\t value of l1 regularization\n";
        cout<<"-loss:\t loss function type\n\t\tHinge|Logit|Square\n";
        cout<<"-opt:\t optimization method:\n\t\tSGD|STG|RDA|FOBOS|Ada-RDA|ADa-FOBOS|AROW\n";
        cout<<"-passes:\t number of passes\n";
        cout<<"-rou:\t rou\n";
        cout<<"-theta:\t value of truncated threshold\n";

        cout<<"\n";
        exit(0);
    }
}
