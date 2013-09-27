/*************************************************************************
  > File Name: Params.cpp
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: Thu 26 Sep 2013 05:49:18 PM SGT
  > Functions: Class for Parsing parameters
 ************************************************************************/
#include "Params.h"
#include "common/md5.h"
#include "common/util.h"

#include <cstdlib>


namespace SOL
{
    void ToUpperCase(string &str)
    {
        string dst_str;
        int len = str.length();
        for (int i = 0; i < len; i++)
            dst_str.push_back(toupper(str[i]));
        std::swap(str,dst_str);
    }

    void ToLowerCase(string &str)
    {
        string dst_str;
        int len = str.length();
        for (int i = 0; i < len; i++)
            dst_str.push_back(tolower(str[i]));
        std::swap(str,dst_str);
    }

    Params::Params(int argc, char** args)
    {
        this->Parse(argc, args);
    }

    void Params::Parse(int argc, char** args)
    {
        this->Default();

        for (int i = 0; i < argc; i++)
        {
            if (strcmp(args[i],"-i") == 0 && i + 1 < argc)
                this->fileName = args[i + 1];
            else if (strcmp(args[i],"-c") == 0 && i + 1 < argc)
                this->cache_fileName = args[i + 1];
            else if (strcmp(args[i],"-t") ==  0 && i + 1 < argc)
                this->test_fileName = args[i+1];
            else if (strcmp(args[i], "-il") == 0 && i + 1 < argc)
                this->labelFile = args[i+1];
            else if (strcmp(args[i], "-tl") == 0 && i + 1 < argc)
                this->test_label_fileName = args[i+1];
            else if (strcmp(args[i],"-tc") == 0 && i + 1 < argc)
                this->test_cache_fileName = args[i + 1];
            else if (strcmp(args[i],"-p") == 0 && i + 1 < argc)
                this->passNum = atoi(args[i+1]);
            else if (strcmp(args[i],"-l1") == 0 && i + 1 < argc)
                this->lambda = strtod(args[i + 1],NULL);
            else if (strcmp(args[i],"-k") == 0 && i + 1 < argc)
                this->K = atoi(args[i + 1]);
            else if (strcmp(args[i],"-eta") == 0 && i + 1 < argc)
                this->eta = strtod(args[i+1],NULL);
            else if (strcmp(args[i],"-theta") == 0 && i + 1 <argc)
                this->theta = strtod(args[i+1],NULL);
            else if (strcmp(args[i],"-delta") == 0 && i + 1 < argc)
                this->delta = strtod(args[i + 1],NULL);
            else if (strcmp(args[i],"-opt") == 0 && i + 1 < argc) //opti method
                this->ParseOptiMethod(args[i+1]);
            else if (strcmp(args[i],"-dt") == 0 && i + 1 < argc) //data type
                this->ParseDataType(args[i + 1]);
            else if(strcmp(args[i],"-d1") == 0 && i + 1 < argc)
                this->digit_1 = atoi(args[i+1]);
            else if (strcmp(args[i],"-d2") == 0 && i + 1  <argc)
                this->digit_2 = atoi(args[i + 1]);
            else if (strcmp(args[i],"-cs") == 0 && i + 1 < argc)
                this->chunk_size = atoi(args[i+1]);
            else if(strcmp(args[i],"-lt") == 0 && i + 1 < argc)
                this->loss_type = this->GetLossType(args[i+1]);
            else if (strcmp(args[i],"-g") == 0 && i + 1 < argc)
                this->gamma = strtod(args[i+1],NULL);
            else if (strcmp(args[i],"-r") == 0 && i + 1 < argc)
                this->rou = strtod(args[i + 1],NULL);
            else if (strcmp(args[i],"--help") == 0)
                this->Help();
            else
                continue;
            i++; 
        }
        //check if cache file exists
        if (this->cache_fileName.size() == 0)
            this->cache_fileName = "tmp_" + md5(this->fileName);
    }

    //default parameter settings
    void Params::Default()
    {
        this->data_type = DataSet_LibSVM;
        this->opti_method = Opti_RDA;
        this->passNum = 1;
        this->eta = -1;
        this->lambda = 0;
        this->theta = -1;
        this->delta = -1;
        this->K = 10;
        this->digit_1 = -1;
        this->digit_2 = -1;
        this->chunk_size = -1;
        this->gamma = -1;
        this->rou = -1;
        this->loss_type = Loss_Type_Logit;
    }

    void Params::ParseOptiMethod(char *str_method)
    {
        string str = str_method;
        ToUpperCase(str);
        const char* c_str = str.c_str();
        if (strcmp(c_str,"SGD") == 0)
            this->opti_method = Opti_SGD;
        else if (strcmp(c_str,"STG") == 0)
            this->opti_method = Opti_STG;
        else if (strcmp(c_str,"RDA") == 0)
            this->opti_method = Opti_RDA;
        else if (strcmp(c_str,"FOBOS") == 0)
            this->opti_method = Opti_FOBOS;
        else if (strcmp(c_str, "ADARDA") == 0)
            this->opti_method = Opti_Ada_RDA;
        else if (strcmp(c_str, "ADAFOBOS") == 0)
            this->opti_method = Opti_Ada_FOBOS;
        else
        {
            cerr<<"Unrecognized Optimization method!"<<endl;
            exit(0);
        }
    }

    void Params::ParseDataType(char *str_data_type)
    {
        string str = str_data_type;
        ToUpperCase(str);
        const char* c_str = str.c_str();
        if (strcmp(c_str,"LIBSVM") == 0)
            this->data_type = DataSet_LibSVM;
        else if (strcmp(c_str,"MNIST") == 0)
            this->data_type = DataSet_MNIST;
        else
        {
            cout<<"Unrecognized Dataset Type!"<<endl;
            exit(0);
        }
    }

    enum_Loss_Type Params::GetLossType(char *str_type)
    {
        string str = str_type;
        ToLowerCase(str);
        const char* c_str = str.c_str();
        if (strcmp(c_str,"hinge") == 0)
            this->loss_type = Loss_Type_Hinge;
        else if (strcmp(c_str,"logit") == 0)
            this->loss_type = Loss_Type_Logit;
        else if (strcmp(c_str,"square") == 0)
            this->loss_type = Loss_Type_Square;
        else
        {
            cout<<"Unrecognized Loss Function Type!"<<endl;
            exit(0);
        }
        return  this->loss_type;

    }

    void Params::Help()
    {
        cout<<"test -i input_data [-c cache_file] [option | value]\n";
        cout<<"Options: \n";
        cout<<"-il:\t input label file (MINIST)\n";
        cout<<"-c:\t cache file name\n";
        cout<<"-t:\t test file name\n";
        cout<<"-tl:\t test label file (MINST)\n";
        cout<<"-tc:\t test cache file (MINST)\n";
        cout<<"-p:\t number of passes\n";
        cout<<"-l1:\t value of l1 regularization\n";
        cout<<"-k:\t number of K in truncated gradient\n";
        cout<<"-eta:\t learning rate\n";
        cout<<"-theta:\t value of truncated threshold\n";
        cout<<"-delta:\t value of delta for Adaptive algorithms\n";
        cout<<"-opt:\t optimization method:\n\t\tSGD|STG|RDA|FOBOS|Ada-RDA|ADa-FOBOS\n";
        cout<<"-dt:\t data type: \n\t\tLibSVM | MNIST\n";
        cout<<"-d1:\t digit value in MNIST\n";
        cout<<"-d2:\t digit value in MNIST\n";
        cout<<"-cs:\t number of chunks for buffering\n";
        cout<<"-lt:\t loss function type\n\t\tHinge|Logit|Square\n";
        cout<<"-g\t gamma\n";
        cout<<"-r\t rou\n";
        exit(0);
    }
}
