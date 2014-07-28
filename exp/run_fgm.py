#This script is to run experiment automatically to test the performance of the algorithm

import os
import time
import sys
import re

import util

def run(dataset, model_config, output_file):
    if util.get_platform() == 'Windows':
        exe_dir = '../extern/FGM_V2/x86/'
        train_exe = exe_dir + 'FGM.exe'
        test_exe = exe_dir + 'Predict.exe'
    else:
        exe_dir = '../extern/fgm/windows/'
        raise Exception('path to liblinear in linux is not set yet!')

    #get the dimension of the data
    data_dim = dataset.dim

    sel_feat_num_list = dataset.l0_list

    dst_folder = dataset.name + '/FGM'
    if os.path.exists(dst_folder) == False:
        os.makedirs(dst_folder)

    result = util.ResultItem()

    for sel_feat_num in sel_feat_num_list:
        model_file =  dst_folder + '/model_%d.txt' %(sel_feat_num)
        predict_file   = dst_folder + '/predict_%d.txt' %(sel_feat_num)
        result_file   = dst_folder + '/result_%d.txt' %(sel_feat_num)
        test_file   = dst_folder + '/test_%d.txt' %(sel_feat_num)

        #clear the file if it already exists
        open(test_file,'w').close()

        result_once = util.ResultItem()

        #evaluate the result
        train_cmd = train_exe + ' -s 12 -c 10 -B %d' %sel_feat_num + ' %s' %dataset.get_train_file(model_config['rand_num']) + ' %s' %model_file 
        train_cmd = train_cmd.replace('/',os.sep)

        print train_cmd
        start_time =time.time()
        os.system(train_cmd)
        end_time = time.time()

        #hard to evaluate train_error, set to zero
        result_once.append_value('train_error',0)

        #parse learning time
        train_time = (float)(end_time - start_time) 
        result_once.append_value('train_time',train_time)

        #predict
        test_cmd = test_exe + ' %s' %dataset.test_file + ' %s' %model_file + ' %s' %predict_file + '>> %s' %test_file
        test_cmd = test_cmd.replace('/',os.sep)
        print test_cmd
        start_time =time.time()
        os.system(test_cmd)
        end_time = time.time()
        test_time = (float)(end_time - start_time) 

        result_once.append_value('test_time',test_time)

        test_error = parse_test_error_rate(test_file)
        result_once.append_value('test_error',test_error)

        result_once.append_value('non_zero_num',sel_feat_num)

        sparse_rate = 100.0 - sel_feat_num * 100.0 / dataset.dim
        result_once.append_value('sparse_rate', sparse_rate)

        result_once.save_result(result_file)

        result.Append(result_once)

        print '\nTraining Result: '
        result_once.Display()
        print '\n'

    result.save_result(output_file)
    return result

def parse_test_error_rate(filename):
    dec_pattern = "(\d+\.?\d*)"
    err_pattern = re.compile(r'Accuracy = ' + dec_pattern) 

    #parse error rate
    with open(filename,'r') as fh:
        content = fh.read()
        err_rate = (float)(err_pattern.findall(content)[0])
        err_rate = (float)('%.2f' %err_rate) 
        if err_rate == None:
            raise ValueError('parse learning error rate failed')
    return 100 - err_rate

