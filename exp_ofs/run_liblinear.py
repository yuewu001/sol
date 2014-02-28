#This script is to run experiment automatically to test the performance of the algorithm

import os
import time
import liblinear_util
import run_util
import exe_path
import l1_def
import sys

def run_liblinear(train_file, test_file,ds, ori_result_file):
    result_all = []

    train_exe_name = exe_path.liblinar_train_exe_name 
    test_exe_name = exe_path.liblinar_test_exe_name 

    #make the result dir
    dst_folder = './%s' %ds
    run_util.create_dir(dst_folder)

    c_list = l1_def.get_lambda_list(ds,'liblinear')

    for c in c_list:
        result_once = [0,0,0,0]
        model_file = dst_folder + '/ll_model%g' %c
        predict_file   = dst_folder + '/ll_predict%g' %c


        result_file = dst_folder + '/' + ori_result_file + '_%f' %c
        #clear the file if it already exists
        open(result_file,'w').close()

        #evaluate the result
        train_cmd = train_exe_name + ' -s 5 -c %f' %c + ' %s' %train_file + ' %s' %model_file 
        test_cmd = test_exe_name + ' %s' %test_file + ' %s' %model_file + ' %s' %predict_file + '>> %s' %result_file
        #test_cmd = test_exe_name + ' %s' %test_file + ' %s' %model_file + ' %s' %predict_file 
        
        print train_cmd
        start_time =time.time()
        os.system(train_cmd)
        end_time = time.time()

        #parse learning time
        train_time = (float)(end_time - start_time) 
        result_once[3] = train_time
        
        #predict
        print test_cmd
        os.system(test_cmd)
        result_once[1] = liblinear_util.parse_error_rate(result_file)
        valid_dim = run_util.get_valid_dim(train_file)
        model_size = liblinear_util.get_model_size(model_file)
        result_once[2] = model_size

        result_all.append(result_once)
    return result_all
