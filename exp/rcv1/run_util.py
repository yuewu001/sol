#This script is some utilities to run experiment automatically 
#to test the performance of the algorithm

import sys
import os
import re

from l1_def import *

exe_name = '../../SOL'

def best_param(cmd_params, opt_name):
    output_file = 'tmp.txt'
    #select the best learning rate
    cmd = exe_name + ' -opt %s' %opt_name
    cmd += cmd_params
    cmd += ' > %s' %output_file 
    print 'learn best parameter...'
    print cmd
    os.system(cmd)
    
    #parse the file to find out the best learning rate
    eta_list = re.findall(r'Best Parameter:\s*eta\s*=\s*.*\n',open(output_file,'r').read())
    if len(eta_list) > 1:
        print 'incorrect result file'
        print eta_list
        sys.exit()
    elif len(eta_list) == 1:
        try:
            best_eta = eta_list[0].split('=')[1].strip()
            best_eta = float(best_eta)
        except ValueError as e:
            print 'convert %s' %best_eta + ' to float value failed!'
            sys.exit()
        print 'best eta = {0}'.format(best_eta)  
    else:
        best_eta = 0

    os.system('rm -f %s' %output_file)
    return best_eta
    
#parse the result to a format for matlab to recognize
def parse_result(input_file, output_file):
    print input_file
    dec_pattern = "(\d+\.?\d*)"
    pattern_l = re.compile(r'Learn error rate:\s*' + dec_pattern + '.*\s*')
    pattern_t = re.compile(r'Test error rate:\s*' + dec_pattern + '.*\s*')
    pattern_s = re.compile(r'Sparsification Rate:\s*' + dec_pattern + '.*\s*')
    pattern_lt= re.compile(r'Learning time:\s*' +dec_pattern + '\s*s') 
    
    result_l = pattern_l.findall(open(input_file,'r').read())
    result_t = pattern_t.findall(open(input_file,'r').read())
    result_s = pattern_s.findall(open(input_file,'r').read())
    result_lt = pattern_lt.findall(open(input_file,'r').read())

    if len(result_t) == 0:
        result_t = result_l
    
    result_list = []
    num = len(result_l)
    for i in range(0,num):
        item = []
        item.append(result_l[i])
        item.append(result_t[i])
        item.append(result_s[i])
        item.append(result_lt[i])
        result_list.append(item)

    open(output_file,'w').close()
    print 'write parsed result %s\n' %output_file
    try:
        file_handler = open(output_file,'w')
        for item in result_list:
            for val in item:
                file_handler.write(str(val) + ' ')
            file_handler.write('\n')
    except IOError as e:
        print "I/O error ({0}): {1}".format(e.errno,e.strerror)
        sys.exit()
    else:
        file_handler.close()

def get_valid_dim(trainfile):
    filename = trainfile + '_info.txt'
    print filename
    dim = 0
    pattern = re.compile(r'valid dim\s*:\s*(\d+)')
    result_list = pattern.findall(open(filename,'r').read())
    if len(result_list) != 1:
        print result_list
        print 'parse failed'
        sys.exit()
    dim = (int)(result_list[0])
    
    return dim
