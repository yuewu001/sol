#!/usr/bin/env python

import sys
import os
import re

thresh = 1e-5
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
    err_pattern = re.compile(r'Accuracy = ' + dec_pattern) 

    #parse error rate
    err_rate = (float)(err_pattern.findall(open(input_file,'r').read())[0])
    err_rate = (float)('%.2f' %err_rate) 
    if err_rate == None:
        print 'parse learning error rate failed'
        sys.exit()
    return err_rate

