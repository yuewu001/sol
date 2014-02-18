#!/usr/bin/env python

import sys
import re

#get the size of a readable model
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

#write the result to file
def write_parse_result(result_list,parse_file):
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

def parse_error_rate(input_file):
    dec_pattern = "(\d+\.?\d*)"
    err_pattern = re.compile(r'average loss = ' + dec_pattern) 

    #parse error rate
    err_rate = (float)(err_pattern.findall(open(input_file,'r').read())[0]) * 100
    err_rate = (float)('%.2f' %err_rate) 
    if err_rate == None:
        print 'parse learning error rate failed'
        sys.exit()
    return err_rate

