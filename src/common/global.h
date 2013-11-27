/*************************************************************************
	> File Name: global.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 8/20/2013 Tuesday 11:16:51 AM
	> Functions: Globally shared definitions
 ************************************************************************/

#ifndef HEADER_GLOBAL_DEFINE
#define HEADER_GLOBAL_DEFINE

#include <stdint.h>
namespace SOL {

#define IndexType uint32_t
    //compress cache
#define BASIC_IO 0
#define GZIP_IO 1
#define ZLIB_IO 2

    enum enum_Loss_Type {
        Loss_Type_Hinge = 0,
        Loss_Type_Logit = 1,
        Loss_Type_Square = 2,
        Loss_Type_SquareHinge = 3,
    };

    enum enum_DataSet_Type {
        //dataset for binary classification
        DataSet_Type_BC = 0x00000100,
        //dataset 0or multiclass classi0ication
        DataSet_Type_MC = 0x00000200,
        //LibSVM DataSet
        DataSet_LibSVM  = 0x00000001,
        //MNISTDATASET
        DataSet_MNIST   = 0x00000002,

        //clear bc/mc
        DataSet_Work_Type_Clear = 0xffff00ff,
        //clear data type
        DataSet_Data_Type_Clear = 0xffffff00,
    };

    //optimization method
    enum enum_Opti_Method {
        Opti_SGD        = 0,
        Opti_STG        = 1,
        Opti_RDA        = 2,
        Opti_RDA_E		= 3,
        Opti_FOBOS      = 4,
        Opti_Ada_RDA    = 5,
        Opti_Ada_FOBOS  = 6,
        Opti_DAROW      = 7,
        Opti_SSAROW     = 8,
        Opti_ASAROW     = 9,
		Opti_CW_RDA		= 10,
    };

    enum NormType {
        NormType_None, //no regularization
        NormType_L1, //L1 regularization
        NormType_L2, //L2 regularization
        NormType_L2S, //L2 squared regularization
    };
}

#endif
