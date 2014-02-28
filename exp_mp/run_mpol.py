#This script is to run experiment automatically to test the performance of the algorithm

import sys
import os

import run_util

buf_size_list = [64,128,256,512,1024,2048,4096,9182]
#buf_size_list = [32]
def run_mpol(opt_name,result_file, dataset, extra_cmd):
    #clear the file if it already exists
    open(result_file,'w').close()
    
    #evaluate the result
    cmd_prefix = run_util.exe_name + extra_cmd + ' -opt %s' %opt_name 
    cmd_postfix = ' >> %s' %result_file
    
    for buf_size in buf_size_list:
        cmd = cmd_prefix + ' -mbs %d' %buf_size + cmd_postfix
        print cmd
        os.system(cmd)
