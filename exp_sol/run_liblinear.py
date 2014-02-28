#This script is to run experiment automatically to test the performance of the algorithm

import os
import time
import liblinear_util
import run_util
import exe_path

def run_liblinear(train_file, test_file,ds, result_file):

    result_once = [0,0,0,0]
    os.system('mkdir ./tmp')

    dst_folder = './%s' %ds
    tmp_folder = dst_folder + '/liblinear'
    #os.system('mkdir -p %s' %tmp_folder)
    run_util.create_dir(tmp_folder)

    model_file = tmp_folder + '/model'
    tmp_file   = tmp_folder + '/tmp.txt'

    train_exe_name = exe_path.liblinar_train_exe_name 
    test_exe_name = exe_path.liblinar_test_exe_name 

    #make the result dir
    #cmd = 'mkdir -p %s' %dst_folder
    #os.system(cmd)
    run_util.create_dir(dst_folder)


    result_file = './%s' %tmp_folder + '/' + result_file
    #clear the file if it already exists
    open(result_file,'w').close()

   
    #evaluate the result
    train_cmd = train_exe_name + ' %s' %train_file + ' %s' %model_file 
    test_cmd = test_exe_name + ' %s' %test_file + ' %s' %model_file + ' %s' %tmp_file + '>> %s' %result_file
    
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
    result_once[2] = 100 - (model_size * 100.0 / valid_dim)

    return [result_once]
