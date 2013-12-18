#This script is to run experiment automatically to test the performance of vw

import sys
import os
import time
import l1_def
import dataset
import run_util
import vw_util

exe_name = 'vw'

def run_vw(train_file, test_file,ds,result_file, is_cache = True):
    tmp_folder = ds + '/vw_tmp'
    os.system('mkdir -p %s' %tmp_folder)
    
    model_file = tmp_folder + '/vw_model'
    rd_model_file = tmp_folder + '/vw_model.txt'
    tmp_file = tmp_folder + '/vw_tmp.txt'
    result_file = tmp_folder + '/' + result_file
    
    extra_cmd = ' --sgd --binary --loss_function=hinge '
    model_cmd = ' --readable_model %s' %rd_model_file + ' -f %s ' %model_file
    
    valid_dim = run_util.get_valid_dim(train_file)
    
    #transform into vw format
    if os.path.exists('%s.vw' %train_file) == False:
        os.system('python ../tools/libsvm2vw.py %s' %train_file)
    if os.path.exists('%s.vw' %test_file) == False:
        os.system('python ../tools/libsvm2vw.py %s' %test_file)
    
    if is_cache == True:
        cache_train = train_file + "_cache.vw"
        cache_test = test_file + "_cache.vw"
    
    train_file += ".vw"
    test_file += ".vw"
    
    #evaluate the result
    if is_cache == True:
        train_cmd_prefix = '%s' %exe_name + ' %s' %train_file +' --cache_file %s ' %cache_train
        test_cmd_prefix = '%s'  %exe_name + ' %s' %test_file + ' -t -i %s' %model_file + ' --cache_file %s ' %cache_test
    else:
        train_cmd_prefix = '%s' %exe_name + ' %s' %train_file 
        test_cmd_prefix = '%s'  %exe_name + ' %s' %test_file + ' -t -i %s' %model_file 
    
    cmd_postfix = ' 2> %s' %tmp_file
    
    result_list = []
    
    
    lambda_list = l1_def.get_lambda_list(ds,'vw')
    for l1 in lambda_list:
        result_item = [0,0,0,0]
        #train
        cmd = train_cmd_prefix + ' --l1 %e' %l1 + extra_cmd + model_cmd +  cmd_postfix
        print cmd
        start_time =time.time()
        os.system(cmd)
        end_time = time.time()
        #parse learning time
        result_item[3] = (float)(end_time - start_time) 
        result_item[3] = (float)('%.2f' %result_item[3]) 
    
        #parse learn error rate
        result_item[0] = vw_util.parse_error_rate(tmp_file)
        #test
        cmd = test_cmd_prefix + extra_cmd + cmd_postfix
        print cmd
        os.system(cmd)
    
        result_item[1] = vw_util.parse_error_rate(tmp_file)
    
        #parse sparsity
        model_size = vw_util.get_model_size(rd_model_file)
        result_item[2] = 100 - (model_size * 100.0 / valid_dim)
    
        result_list.append(result_item)
    
    vw_util.write_parse_result(result_list,result_file)
    return result_list
