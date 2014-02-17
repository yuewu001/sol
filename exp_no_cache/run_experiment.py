#This script is to run experiment automatically to test the performance of the algorithm

import sys
import os

import run_util
import l1_def

def run_experiment(opt_name,result_file, dataset, extra_cmd):
    #clear the file if it already exists
    open(result_file,'w').close()
    
    #evaluate the result
    cmd_prefix = run_util.exe_name + extra_cmd + ' -opt %s' %opt_name 
    cmd_postfix = ' >> %s' %result_file

    if 'synthetic' in dataset:
        bs_list = l1_def.get_lambda_list(dataset, opt_name)
    elif opt_name == 'liblinear':
        bs_list = l1_def.get_lambda_list(dataset, opt_name)
    else:
        temp_list = extra_cmd.split()
        train_file = ''
        for k in range(0,len(temp_list)):
            if temp_list[k] == '-i':
                train_file = temp_list[k+1]
                break;
        if train_file == '':
            print 'no input file is specified!'
            sys.exit()
        data_valid_dim = run_util.get_valid_dim(train_file)
        print data_valid_dim
        lambda_list = l1_def.get_lambda_list(dataset, opt_name)

        bs_list = [] 
        b_num = len(lambda_list)
        for i in range(0,b_num):
            dim = data_valid_dim * (1 - lambda_list[i])
            if dim > 0:
                bs_list.append(dim)

    for bs in bs_list:
        cmd = cmd_prefix + ' -k %d' %bs + cmd_postfix
        print cmd
        os.system(cmd)
