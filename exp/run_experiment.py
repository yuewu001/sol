#This script is to run experiment automatically to test the performance of the algorithm

import sys
import os

import run_util
import l1_def

def Usage():
    print 'Usage: run_experiment.py opt_name dst_folder dataset [parameters]'

if len(sys.argv) < 3:
    Usage()
    sys.exit()

#parse the arguments
opt_name = sys.argv[1]
dst_folder = sys.argv[2]
dataset = sys.argv[3]

is_best_param = False
is_l1 = True
extra_cmd = ' '
for k in range(4,len(sys.argv)):
    if sys.argv[k] == 'no_l1':
        is_l1 = False
        continue
    if sys.argv[k] == '-lbp':
        is_best_param = True

    extra_cmd = extra_cmd + sys.argv[k] + ' '

result_file = dst_folder + '/%s' %opt_name + '_result.txt'
#clear the file if it already exists
open(result_file,'w').close()

#evaluate the result
cmd_prefix = run_util.exe_name + extra_cmd + ' -opt %s' %opt_name 
cmd_postfix = ' >> %s' %result_file

#learn the best parameter
if is_best_param == True:
    output_file = dst_folder + '/%s_best.txt' %opt_name
    run_util.best_param(extra_cmd, opt_name, output_file) 
    sys.exit()

if opt_name == 'ASAROW':
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

if is_l1 == True:
    lambda_list = l1_def.get_lambda_list(dataset, opt_name)
    for l1 in lambda_list:
        if opt_name == 'ASAROW':
            l1 = (int)(l1 * data_valid_dim)
	    if l1 < 1:
		continue
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
run_util.parse_result(result_file, parse_file);
