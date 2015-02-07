#!/usr/bin/env python
# AUTHOR:   Yue Wu (yuewu@outlook.com)
# FILE:     sol_rand_split.py
# ROLE:     split a file randomly
# CREATED:  2015-02-07 20:50:07
# MODIFIED: 2015-02-07 21:01:59
r"""used to split a  dataset to two parts randomly
"""
import sys
import random

def Usage():
    print 'sol_sample.py in_file out_file1 line_num1 out_file2 line_num2'
    sys.exit()

if len(sys.argv) != 6:
    Usage()

in_filename = sys.argv[1]
out_filename1 = sys.argv[2]
line_num1 = int(sys.argv[3])
out_filename2 = sys.argv[4]
line_num2 = int(sys.argv[5])

line_num_sum = line_num1 + line_num2
line_num = 0

try:
    in_file = open(in_filename,"rb")
    out_file1 = open(out_filename1,"wb")
    out_file2 = open(out_filename2,"wb")
    while True:
        line = in_file.readline()
        if len(line) == 0:
            break;
        line_num += 1
        if line_num % 1000 == 0:
            sys.stdout.write('line number	: %d\r' %(line_num))
        if  random.randint(1,line_num_sum) <= line_num1:
            out_file1.write(line)
        else:
            out_file2.write(line)

except IOError as e:
    print "I/O error ({0}: {1})".format(e.errno, e.stderror)
    in_file.close()
    out_file1.close()
    out_file2.close()
    sys.exit()
else:
    in_file.close()
    out_file1.close()
    out_file2.close()

print 'line number	: %d' %line_num
