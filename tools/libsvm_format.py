#!/usr/bin/env python
"""format a libsvm dataset, change label 0 to -1,""" 
import sys
import os
import time

def Usage():
    print 'Usage: libsvm_format.py filename output_filename'

if len(sys.argv) != 3:
    Usage()
    sys.exit()

filename = sys.argv[1]
output_filename = sys.argv[2]

start = time.time()
print 'change label 0 to -1'
cmd = 'cat %s' %filename + " | sed -e 's/^0/-1/' " + " > %s" %output_filename
os.system(cmd)
#transform to 
end = time.time()
print('time elapsed: ' + str(end -start))
