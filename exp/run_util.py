#This script is some utilities to run experiment automatically 
#to test the performance of the algorithm

import sys
import os
import re

from l1_def import *

exe_name = '../SOL'

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
    dec_pattern = "(\d+\.?\d*)"
    pattern = re.compile(r'Learn error rate:\s*' + dec_pattern + '.*\s*' +
            'Test error rate:\s*' + dec_pattern + '.*\s*' +
            'Sparsification Rate:\s*' + dec_pattern + '.*\s*' +
            'Learning time:\s*' +dec_pattern + '\s*s') 
    
    result_list = pattern.findall(open(input_file,'r').read())
    
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
