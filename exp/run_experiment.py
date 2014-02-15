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
    
    bs_list = l1_def.get_lambda_list(dataset, opt_name)

    for bs in bs_list:
        cmd = cmd_prefix + ' -k %d' %bs + cmd_postfix
        print cmd
        os.system(cmd)
