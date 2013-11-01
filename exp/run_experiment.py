#This script is to run experiment automatically to test the performance of the algorithm

import sys
import os
import re

from run_util import *
import l1_def

def Usage():
    print 'Usage: run_experiment.py opt_name dst_folder [parameters]'

if len(sys.argv) < 3:
    Usage()
    sys.exit()

#parse the arguments
opt_name = sys.argv[1]
dst_folder = sys.argv[2]

is_best_param = True
is_l1 = True
extra_cmd = ' '
for k in range(3,len(sys.argv)):
    if sys.argv[k] == 'no_l1':
        is_l1 = False
        continue
    if sys.argv[k] == '-eta':
        is_best_param = False
    extra_cmd = extra_cmd + sys.argv[k] + ' '

result_file = dst_folder + '/%s' %opt_name + '_result.txt'
#clear the file if it already exists
open(result_file,'w').close()

#evaluate the result
cmd_prefix = exe_name + extra_cmd + ' -opt %s' %opt_name 
cmd_postfix = ' >> %s' %result_file

#learn the best parameter
if is_best_param == True:
    best_eta = best_param() 
    cmd_prefix += ' -eta %e' %best_eta

if is_l1 == True:
    lambda_list = l1_def.get_lambda_list(opt_name)
    for l1 in lambda_list:
        if opt_name == 'ASAROW':
            l1 = (int)(l1 * 47152)
            cmd = cmd_prefix + ' -k %d' %l1 + cmd_postfix
        else:
            cmd = cmd_prefix + ' -l1 %e' %l1 + cmd_postfix
        print cmd
        os.system(cmd)
else:
    cmd = cmd_prefix + cmd_postfix
    print cmd
    os.system(cmd)

print '\nparsing result...'
#write the result to file
parse_file = dst_folder +'/%s' %opt_name + '.txt'
parse_result(result_file, parse_file);
