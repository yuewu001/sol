#This script is to run experiment automatically to test the performance of the algorithm

import sys
import os
import re

def Usage():
    print 'Usage: run_experiment.py opt_name'

exe_name = './test'
if len(sys.argv) < 2:
    Usage()
    sys.exit()

extra_cmd = ' '
for k in range(2,len(sys.argv)):
    extra_cmd = extra_cmd + sys.argv[k] + ' '

lambda_start = 1e-8
lambda_end = 10
lambda_step = 10

#make the result dir
cmd = 'mkdir -p ./result'
os.system(cmd)

opt_name = sys.argv[1]
result_file = './result/%s' %opt_name + '_result.txt'
print 'Algorithm: ' + opt_name 
print 'output file %s' %result_file
#clear the file if it already exists
open(result_file,'w').close()

#select the best learning rate
cmd = exe_name + ' -opt %s' %opt_name
cmd += extra_cmd
cmd += ' >> %s' %result_file
print 'lear best parameter...'
print cmd
os.system(cmd)

#open the result file
try:
    file_handler = open(result_file,'r')
    file_content = file_handler.readlines()
except IOError as e:
    print "I/O error ({0}): {1}".format(e.errno,e.strerror)
    sys.exit()
else:
    file_handler.close()

#parse the file to find out the best learning rate
eta_list = re.findall(r'Best Parameter:\s*eta\s*=\s*.*\n',open(result_file,'r').read())
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

#evaluate the result
l1 = lambda_start
cmd_prefix = exe_name + ' -opt %s' %opt_name + ' -eta %e' %best_eta
cmd_postfix = ' >> %s' %result_file
while l1 <= lambda_end:
    cmd = cmd_prefix + ' -l1 %e' %l1 + extra_cmd +  cmd_postfix
    print cmd
    os.system(cmd)
    l1 *= lambda_step

print 'parsing result...'
#parse the result to a format for matlab to recognize
dec_pattern = "(\d+\.?\d*)"
pattern = re.compile(r'Learn error rate:\s*' + dec_pattern + '.*\s*' +
        'Test error rate:\s*' + dec_pattern + '.*\s*' +
        'Sparsification Rate:\s*' + dec_pattern + '.*\s*' +
        'Learning time:\s*' +dec_pattern + '\s*s') 

result_list = pattern.findall(open(result_file,'r').read())

#write the result to file
parse_file = './result/%s' %opt_name + '.txt'
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
