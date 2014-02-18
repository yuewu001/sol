#This script is to run experiment automatically to test the performance of the algorithm

import os
import time
import liblinear_util
import run_util
import exe_path
import l1_def

def run_fgm(train_file, test_file,ds, ori_result_file):
    result_all = []

    train_exe_name = exe_path.fgm_train
    test_exe_name = exe_path.fgm_test

    #make the result dir
    dst_folder = './%s' %ds
    run_util.create_dir(dst_folder)

    if 'synthetic' in ds:
        bs_list = l1_def.get_lambda_list(ds,'fgm')
    else:
        data_valid_dim = run_util.get_valid_dim(train_file)
        lambda_list = l1_def.get_lambda_list(ds,'fgm')

        bs_list = []
        b_num = len(lambda_list)
        for i in range(0,b_num):
            dim = data_valid_dim * (1 - lambda_list[i])
            if dim > 0:
                bs_list.append(dim)

    for bs in bs_list:
        result_once = [0,0,0,0]
        model_file = dst_folder + '/fgm_model%g' %bs
        predict_file   = dst_folder + '/fgm_predict%g' %bs


        result_file = dst_folder + '/' + ori_result_file + '_%d' %bs
        #clear the file if it already exists
        open(result_file,'w').close()

        #evaluate the result
        train_cmd = train_exe_name + ' -s 12 -c 10 -B %d' %bs + ' %s' %train_file + ' %s' %model_file 
        test_cmd = test_exe_name + ' %s' %test_file + ' %s' %model_file + ' %s' %predict_file + '>> %s' %result_file
        
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
        result_once[2] = bs

        result_all.append(result_once)
    return result_all
