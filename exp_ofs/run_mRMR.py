#This script is to run experiment automatically to test the performance of the algorithm

import os
import time
import liblinear_util
import run_util
import exe_path
import l1_def
import re
import dataset

import sys

def parse_model_file(model_file,parse_file):
    print 'parse model file of mRMR%s\n' %model_file
    c_feat = []
    pattern = re.compile(r'(\S*)\s*')
    is_begin = False
    try:
        file_handler = open(model_file,'r')
        while True:
            line = file_handler.readline()
            line = line.strip()
            if is_begin == True and len(line) == 0:
                break
            if line == '*** mRMR features ***':
                line = file_handler.readline()
                is_begin = True
                continue
            if (is_begin == False):
                continue
            result_list = pattern.findall(line)
            c_feat.append(int(result_list[1]))
    except IOError as e:
        print "I/O error ({0}): {1}".format(e.errno,e.strerror)
        sys.exit()
    else:
        file_handler.close()
        print 'feature number %d' %(len(c_feat))
    #write c_feat into file
    try:
        file_handler = open(parse_file,'wb')
    
        for k in range(0,len(c_feat)):
            file_handler.write('%d\n' %c_feat[k])
    except IOError as e:
        print "I/O error ({0}): {1}".format(e.errno,e.strerror)
        sys.exit()
    else:
        file_handler.close()
    return c_feat
    

def run_mRMR(train_file, test_file,ds, result_file):
    result_all = []

    train_exe_name = exe_path.mRMR

    #make the result dir
    dst_folder = './%s' %ds
    run_util.create_dir(dst_folder)

    data_valid_dim = run_util.get_valid_dim(train_file)
    data_num = run_util.get_data_num(train_file)

    #bs_list = l1_def.get_lambda_list(ds,'mRMR')
    
    if 'synthetic' in ds:
        bs_list = l1_def.get_lambda_list(ds,'mRMR')
    else:
        lambda_list = l1_def.get_lambda_list(ds,'mRMR')

        bs_list = [] 
        b_num = len(lambda_list)
        for i in range(0,b_num):
            dim = int(data_valid_dim * (1 - lambda_list[i]))
            if dim > 0 and dim <= 500:
                bs_list.append(dim)

    bs_list = l1_def.get_lambda_list(ds,'mRMR')

    #clear the file if it already exists
    open(result_file,'w').close()

    for bs in bs_list:
        result_once = [0,0,0,0]
        model_file = dst_folder + '/mRMR_model%d' %bs
        parse_file = dst_folder + '/mRMR_model_parse%d' %bs 

        if os.path.exists(model_file) == False:
            print model_file + ' not exist'
            csv_train_file = train_file + '.csv'
            if os.path.exists(csv_train_file) == False:
                #convert data
                print 'convert data'
                cmd = exe_path.csv_converter + ' -i %s' %train_file + ' -o %s' %csv_train_file
                cmd += ' -sdt libsvm -ddt csv'
                print cmd
                os.system(cmd)

            #run mRMR
            prev_cmd = train_exe_name + ' -v %d' %data_valid_dim + ' -t 0.5 -i %s' %csv_train_file 
            cmd = prev_cmd + ' -n %d' %bs + ' > %s' %model_file
            print cmd
            start_time =time.time()
            os.system(cmd)
            end_time = time.time()

            #parse learning time
            train_time = (float)(end_time - start_time) 
            result_once[3] = train_time

            #parse result
            parse_model_file(model_file,parse_file);

        #run OGD
        cmd_data = dataset.get_cmd_data_by_file(train_file, test_file,True)
        cmd = exe_path.SOL_exe_name + cmd_data + ' -m %s' %parse_file + ' -k %d' %bs
        cmd += dataset.get_model_param(ds,'SGD-FS')
        cmd += ' -opt mRMR_OGD -norm -loss Hinge >> %s' %result_file

        print cmd
        os.system(cmd)

        result_once[2] = bs
        
        result_all.append(result_once)
    return result_all
