#!/usr/bin/env python
r"""used to sample dataset to keep balance
"""
import sys
import random

def Usage():
    print 'sol_sample.py filename out_file pattern sample_rate'
    sys.exit()

if len(sys.argv) != 5:
    Usage()

filename = sys.argv[1]
out_filename = sys.argv[2]
pattern = sys.argv[3]
sample_rate = int(sys.argv[4])

try:
    in_file = open(filename,"rb")
    out_file = open(out_filename,"wb")
    while True:
        line = in_file.readline()
        if len(line) == 0:
            break;
        elif line.startswith(pattern):
            if random.randint(1,sample_rate) != sample_rate:
                continue;
        out_file.write(line)

except IOError as e:
    print "I/O error ({0}: {1})".format(e.errno, e.stderror)
    in_file.close()
    out_file.close()
    sys.exit()
else:
    in_file.close()
    out_file.close()
