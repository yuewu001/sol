#This script is to run experiment automatically to test the performance of vw

import sys
import os
import re
import time
import l1_def

from vw_l1_def import *
from run_util import *

#exe_name = '../extern/vw/vw'
exe_name = '~/work/vw/vw'
model_file = './vw_tmp/vw_model'
rd_model_file = './vw_tmp/vw_model.txt'
tmp_file = './vw_tmp/vw_tmp.txt'

os.system('mkdir vw_tmp')

def Usage():
    print 'Usage:run_vw.py dst_folder trainfile cache_train testfile cache_test'


def get_model_size(model_file):
    int_pattern = "(\d+)"
    dec_pattern = "([+-]?\d+\.?\d*)"
    pattern = re.compile(r'' + int_pattern + ':' + dec_pattern) 
    
    model_size = 0
    try:
        file_handle = open(model_file,'r')
        while True:
            line = file_handle.readline()
            if not line:
                break;
    
            if pattern.match(line) != None:
                model_size += 1
    
        #result_list = pattern.findall(open(model_file,'r').read())
    
    except IOError as e:
        print "I/O error ({0}): {1}".format(e.errno,e.strerror)
        sys.exit()
    else:
        file_handle.close()
    
    return model_size

if len(sys.argv) < 4:
    Usage()
    sys.exit()

dst_folder = sys.argv[1]

extra_cmd = ' --passes 5 --sgd --binary --loss_function=logistic --readable_model %s' %rd_model_file + ' -f %s' %model_file
trainfile = sys.argv[2]
cache_train = sys.argv[3]

is_test = False
if (len(sys.argv) > 4):
    testfile = sys.argv[4]
    cache_test = sys.argv[5]
    is_test = True

is_l1 = True
is_cache = True
for k in range(6,len(sys.argv)):
    if sys.argv[k] == 'no_l1':
        is_l1 = False
    if sys.argv[k] == 'no_cache':
        is_cache = False

valid_dim = get_valid_dim(trainfile)

#transform into vw format
if os.path.exists('%s.vw' %trainfile) == False:
    os.system('python ../tools/libsvm2vw.py %s' %trainfile)
if is_test == True:
    if os.path.exists('%s.vw' %testfile) == False:
        os.system('python ../tools/libsvm2vw.py %s' %testfile)

trainfile += ".vw"
cache_train += ".vw"
if is_test == True:
    testfile += ".vw"
    cache_test += ".vw"

#make the result dir
cmd = 'mkdir -p ./%s' %dst_folder
os.system(cmd)

print '----------------------------------------\nAlgorithm: vw'

#select the best learning rate
cmd = exe_name
cmd += extra_cmd

#evaluate the result
if is_cache == True:
    train_cmd_prefix = '%s' %exe_name + ' %s' %trainfile +' --cache_file %s ' %cache_train
    if is_test == True:
        test_cmd_prefix = '%s'  %exe_name + ' %s' %testfile + ' -t -i %s' %model_file + ' --cache_file %s ' %cache_test
else:
    train_cmd_prefix = '%s' %exe_name + ' %s' %trainfile
    if is_test == True:
        test_cmd_prefix = '%s'  %exe_name + ' %s' %testfile + ' -t -i %s' %model_file 

cmd_postfix = ' 2> %s' %tmp_file

result_list = []

dec_pattern = "(\d+\.?\d*)"
err_pattern = re.compile(r'average loss = ' + dec_pattern) 

l1 = lambda_start
while l1 <= lambda_end:
    result_item = [0,0,0,0]
    #train
    if is_l1 == False:
        cmd = train_cmd_prefix + extra_cmd +  cmd_postfix
    else:
        cmd = train_cmd_prefix + ' --l1 %e' %l1 + extra_cmd +  cmd_postfix
    print cmd
    start_time =time.time()
    os.system(cmd)
    end_time = time.time()
    #parse learning time
    result_item[3] = (float)(end_time - start_time) 
    result_item[3] = (float)('%.2f' %result_item[3]) 

    #parse learn error rate
    result_item[0] = (float)(err_pattern.findall(open(tmp_file,'r').read())[0]) * 100
    result_item[0] = (float)('%.2f' %result_item[0]) 
    if result_item[0] == None:
        print 'parse learning error rate failed'
        sys.exit()
    
    #test
    if is_test == True:
        cmd = test_cmd_prefix + cmd_postfix
        print cmd
        os.system(cmd)

        #parse test error rate
        result_item[1] = (float)(err_pattern.findall(open(tmp_file,'r').read())[0]) * 100
        result_item[1] = (float)('%.2f' %result_item[1]) 
        if result_item[1] == None:
            print 'parse test error rate failed'
            sys.exit()
    else:
        result_item[1] = result_item[0]

    #parse sparsity
    model_size = get_model_size(rd_model_file)
    result_item[2] = 100 - (model_size * 100.0 / valid_dim)

    result_list.append(result_item)

    l1 *= lambda_step
    if is_l1 == False:
        break;

#write the result to file
parse_file = './%s' %dst_folder +'/vw.txt'
open(parse_file,'w').close()
print 'write parsed result %s' %parse_file
try:
    file_handler = open(parse_file,'w')
    for item in result_list:
        for val in item:
            file_handler.write(str(val) + ' ')
        file_handler.write('\n')
except IOError as e:
    print "I/O error ({0}): {1}".format(e.errno,e.strerror)
    sys.exit()
else:
    file_handler.close()
