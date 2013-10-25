#This script is to run experiment automatically to test the performance of the algorithm

import sys
import os
import re
import time

from l1_def import *

def Usage():
    print 'Usage: run_liblinear.py dst_folder model_file train_file test_file' 

if len(sys.argv) < 5:
    Usage()
    sys.exit()

dst_folder = sys.argv[1]
model_file = sys.argv[2]
train_file = sys.argv[3]
test_file  = sys.argv[4]

train_exe_name = './train'
test_exe_name = './predict'

#make the result dir
cmd = 'mkdir -p ./%s' %dst_folder
os.system(cmd)

print 'Algorithm: LibLinear'
result_file = './%s' %dst_folder + '/liblinear_result.txt'
print 'output file %s' %result_file
#clear the file if it already exists
open(result_file,'w').close()

   
#evaluate the result
train_cmd = train_exe_name + ' %s' %train_file + ' %s' %model_file 
test_cmd = test_exe_name + ' %s' %test_file + ' %s' %model_file + ' tmp.txt >> %s' %result_file

print train_cmd
start_time =time.time()
os.system(train_cmd)
end_time = time.time()
#parse learning time
train_time = (float)(end_time - start_time) 
train_time = ('training time: %.2f\n' %train_time) 
file_handler = open(result_file, 'w')
file_handler.write(train_time)
file_handler.close()

#predict
os.system(test_cmd)
