#This script is to run experiment automatically to test the performance of the algorithm

import os
import time
import sys
import re

import util

def run(dataset,model_config):
    if util.get_platform() == 'Windows':
        bif_exe = '../extern/FST3/x64/BIF.exe'
    else:
        bif_exe = '../extern/FST3/linux/BIF'

    converter_py = 'python ../tools/libsvm2arff.py'

    dst_folder = dataset.name + '/BIF'
    if os.path.exists(dst_folder) == False:
        os.makedirs(dst_folder)

    data_dim = dataset.dim
    data_num = dataset.data_num

    #bs_list = l1_def.get_lambda_list(ds,'bif')

    sel_feat_num_list = [x for x in dataset.l0_list] 

    train_file = dataset.train_file
    arff_train_file =  train_file + '.arff'

    for sel_feat_num in sel_feat_num_list:
        model_file = dst_folder + '/model_%d' %sel_feat_num

        #run BIF
        train_time  = 0
        #prepare training data
        if os.path.exists(model_file) == False:
            if os.path.exists(arff_train_file) == False:
                #convert data
                print 'convert data'
                cmd = converter_py + ' \"%s\" \"%s\"' %(train_file, arff_train_file)
                cmd = cmd.replace('/',os.sep)
                print cmd
                os.system(cmd)

            cmd = bif_exe + ' %s %d %s' %(arff_train_file, sel_feat_num, model_file)
            cmd = cmd.replace('/',os.sep)
            print cmd
            start_time =time.time()
            os.system(cmd)
            end_time = time.time()

            #parse learning time
            bif_train_time = (float)(end_time - start_time)

    if os.path.exists(arff_train_file) == True:
        os.remove(arff_train_file)

