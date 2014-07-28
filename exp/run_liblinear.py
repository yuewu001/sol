#This script is to run experiment automatically to test the performance of the algorithm

import os
import time
import sys
import re

import util

def run(dataset, model_config, output_file):
    if util.get_platform() == 'Windows':
        exe_dir = '../extern/liblinear/windows/'
        train_exe = exe_dir + 'train.exe'
        test_exe = exe_dir + 'predict.exe'
    else:
        exe_dir = '../extern/liblinear/windows/'
        raise Exception('path to liblinear in linux is not set yet!')


    c_list = dataset.c_list

    dst_folder = dataset.name + '/liblinear'
    if os.path.exists(dst_folder) == False:
        os.makedirs(dst_folder)

    result = util.ResultItem()

    for c in c_list:
        model_file =  dst_folder + '/model_%g.txt' %(c)
        predict_file   = dst_folder + '/predict_%g.txt' %(c)
        result_file   = dst_folder + '/result_%g.txt' %(c)
        test_file   = dst_folder + '/test_%g.txt' %(c)

        #clear the result file
        open(test_file,'w').close()

        result_once = util.ResultItem()

        #training 
        train_cmd = train_exe + ' -s 5 -c %f' %c + ' %s' %dataset.get_train_file(model_config['rand_num']) + ' %s' %model_file
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

        model_size = get_model_size(model_file)

        result_once.append_value('non_zero_num',model_size)

        sparse_rate = 100.0 - model_size * 100.0 / dataset.dim
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

#get the size of a readable model
def get_model_size(model_file):
    thresh = 1e-5
    model_size = 0
    try:
        file_handle = open(model_file,'r')
        line_num = 0
        while True:
            line = file_handle.readline()
            if not line:
                break;
            line_num += 1
            if line_num < 7:
                continue

            weights = filter(None,line.strip().split(' '))
            if any([abs(float(x)) > thresh for x in weights]):
                model_size += 1 
    except IOError as e:
        print "I/O error ({0}): {1}".format(e.errno,e.strerror)
        sys.exit()
    else:
        file_handle.close()

    return model_size

