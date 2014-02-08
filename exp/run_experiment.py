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
    
    lambda_list = l1_def.get_lambda_list(dataset, opt_name)

    if opt_name == 'AROW-FS' or opt_name == 'SGD-FS' or opt_name == 'OFSGD':
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

    for l1 in lambda_list:
        if opt_name == 'AROW-FS' or opt_name == 'SGD-FS' or opt_name == 'OFSGD':
            l1 = (int)(l1 * data_valid_dim)
            if l1 < 1:
		        continue
            cmd = cmd_prefix + ' -k %d' %l1 + cmd_postfix
        else:
            cmd = cmd_prefix + ' -l1 %e' %l1 + cmd_postfix
        print cmd
        os.system(cmd)
