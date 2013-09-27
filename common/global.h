/*************************************************************************
	> File Name: global.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 8/20/2013 Tuesday 11:16:51 AM
	> Functions: Globally shared definitions
 ************************************************************************/

#pragma once

namespace SOL
{
    enum enum_Loss_Type
    {
        Loss_Type_Hinge = 0,
        Loss_Type_Logit = 1,
        Loss_Type_Square = 2,
    };

    enum enum_DataSet_Type
    {
        //dataset for binary classification
        DataSet_Type_BC = 0x00000100,
        //dataset for multiclass classification
        DataSet_Type_MC = 0x00001000,
        //LibSVM DataSet
        DataSet_LibSVM  = 1,
        //MNISTDATASET
        DataSet_MNIST   = 2,
    };

    //optimization method
    enum enum_Opti_Method
    {
        Opti_SGD        = 0,
        Opti_STG        = 1,
        Opti_RDA        = 2,
        Opti_FOBOS      = 3,
        Opti_Ada_RDA    = 4,
        Opti_Ada_FOBOS  = 5,
    };

    enum NormType
    {
        NormType_None, //no regularization
        NormType_L1, //L1 regularization
        NormType_L2, //L2 regularization
        NormType_L2S, //L2 squared regularization
    };
}
