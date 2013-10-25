#This script is to run experiment automatically to test the performance of vw

import sys
import os
import re
import time

def Usage():
    print 'Usage: run_experiment.py dst_folder trainfile cache_train testfile cache_test'

def get_valid_dim(trainfile):
    filename = trainfile + '_info.txt'
    dim = 0
    pattern = re.compile(r'valid dim\s*:\s*(\d+)')
    result_list = pattern.findall(open(filename,'r').read())
    if len(result_list) != 1:
        print result_list
        print 'parse failed'
        sys.exit()
    dim = (int)(result_list[0])
    
    return dim

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

if len(sys.argv) != 6:
    Usage()
    sys.exit()

dst_folder = sys.argv[1]
exe_name = './vw'

extra_cmd = ' --sgd --binary --loss_function=logistic --readable_model model.txt -f model '
trainfile = sys.argv[2]
cache_train = sys.argv[3]
testfile = sys.argv[4]
cache_test = sys.argv[5]

valid_dim = get_valid_dim(trainfile)

#transform into vw format
if os.path.exists('%s.vw' %trainfile) == False:
    os.system('python libsvm2vw.py %s' %trainfile)
if os.path.exists('%s.vw' %testfile) == False:
    os.system('python libsvm2vw.py %s' %testfile)

trainfile += ".vw"
cache_train += ".vw"
testfile += ".vw"
cache_test += ".vw"

lambda_start = 1e-8
lambda_end = 10
lambda_step = 10

#make the result dir
cmd = 'mkdir -p ./%s' %dst_folder
os.system(cmd)

print '----------------------------------------\nAlgorithm: vw'

#select the best learning rate
cmd = exe_name
cmd += extra_cmd

#evaluate the result
l1 = lambda_start
train_cmd_prefix = '%s' %exe_name + ' %s' %trainfile +' --cache_file %s ' %cache_train
test_cmd_prefix = '%s'  %exe_name + ' %s' %testfile + ' -t -i model --cache_file %s ' %cache_test
cmd_postfix = ' 2> vw_tmp.txt'

result_list = []

dec_pattern = "(\d+\.?\d*)"
err_pattern = re.compile(r'average loss = ' + dec_pattern) 

while l1 <= lambda_end:
    result_item = [0,0,0,0]
    #train
    cmd = train_cmd_prefix + ' --l1 %e' %l1 + extra_cmd +  cmd_postfix
    print cmd
    start_time =time.time()
    os.system(cmd)
    end_time = time.time()
    #parse learning time
    result_item[3] = (float)(end_time - start_time) 
    result_item[3] = (float)('%.2f' %result_item[3]) 

    #parse learn error rate
    result_item[0] = (float)(err_pattern.findall(open('vw_tmp.txt','r').read())[0]) * 100
    result_item[0] = (float)('%.2f' %result_item[0]) 
    if result_item[0] == None:
        print 'parse learning error rate failed'
        sys.exit()
    
    #test
    cmd = test_cmd_prefix + cmd_postfix
    print cmd
    os.system(cmd)

    #parse test error rate
    result_item[1] = (float)(err_pattern.findall(open('vw_tmp.txt','r').read())[0]) * 100
    result_item[1] = (float)('%.2f' %result_item[1]) 
    if result_item[1] == None:
        print 'parse test error rate failed'
        sys.exit()
    #parse sparsity
    model_size = get_model_size('model.txt')
    result_item[2] = 100 - (model_size * 100.0 / valid_dim)

    result_list.append(result_item)

    l1 *= lambda_step

#write the result to file
parse_file = './%s' %dst_folder +'/vw.txt'
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
