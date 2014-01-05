#!/usr/bin/env python
"""Cross validation"""

import re
import sys
import os
import run_util
import dataset

def Usage():
    print 'Usage: CV dataset algorithm fold_num param start:step:end'
    print '\tstep: for multiplication'

if len(sys.argv) < 6 or len(sys.argv) % 2 == 1:
    Usage()
    sys.exit()

exe_cmd = run_util.exe_name

#extra command
exe_cmd += ' -loss Hinge -norm '

dt = sys.argv[1]
opt_name = sys.argv[2]
exe_cmd += ' -opt %s' %opt_name
fold_num = int(sys.argv[3])
if fold_num < 2 or fold_num > 26:
    print 'error: fold number must bigger than 2 and less 26'
    sys.exit()

split_list = dataset.get_cv_data_list(dt,fold_num)

dst_folder = dt + '/cv'
os.system('mkdir -p %s' %dst_folder)

#define the grid search item
class grid_item:
    def __init__(self):
        self.name = ''
        self.start_val = 0
        self.step_val = 0
        self.end_val = 0

    def val(self, index):
        ret = self.start_val
        while index > 0:
            ret *= self.step_val
            index -= 1
        return ret

    def size(self):
        if self.end_val <= self.start_val:
            return 0
        else:
            count = 0 
            val = self.start_val
            while val <= self.end_val:
                val *= self.step_val
                count += 1
            return count

    def __str__(self):
        return 'name: {0} value: {1}:{2}:{3}'\
                .format(self.name,self.start_val, self.step_val,self.end_val)
    

#detect param
param_pattern   = r'(?P<param_name>-\w+)'
num_pattern     = r'\d*\.?\d+'
search_pattern  = r'(?P<start_val>{0}):(?P<step_val>{1}):(?P<end_val>{2})'\
        .format(num_pattern,num_pattern,num_pattern)

param_space = []
k = 4 
while k < len(sys.argv):
    param   = re.match(param_pattern,sys.argv[k])
    k += 1
    search  = re.match(search_pattern, sys.argv[k])
    k += 1
    
    item = grid_item()
    if param and search:
        item.name = param.group('param_name')
        item.start_val  = (float)(search.group('start_val'))
        item.step_val   = (float)(search.group('step_val'))
        item.end_val    = (float)(search.group('end_val'))

        param_space.append(item)
    else:
        print 'incorrect input parameter {0} {1}'.format(sys.argv[k-2],sys.argv[k-1])
        sys.exit()

item_num = len(param_space)
size_list = []
for item in param_space:
    size_list.append(item.size())
grid_size = reduce(lambda x, y: x * y, size_list)

param_list = []

run_count = 0

#run on one split of data
def run_one_data(exe_cmd_one):
    global run_count
    result_file = dst_folder + '/%s' %opt_name + '_result_%d' %run_count + '.txt'
    os.system('rm -f %s' %result_file)
    for k in range(0,grid_size):
        cmd = exe_cmd_one
        param_item = []
        for j in range(0, item_num):
            coor = k % size_list[j]
            k = int(k / size_list[j])
            param_item.append(param_space[j].name)
            param_item.append(param_space[j].val(coor))
            cmd += ' {0} {1}'.format(param_space[j].name, \
                    param_space[j].val(coor)) 
        param_list.append(param_item)
        #write to file
        cmd += ' >> %s' %result_file
        print cmd
        os.system(cmd)
    
    #write the result to file
    parse_file = dst_folder +'/%s' %opt_name + '_parse_%d' %run_count + '.txt'
    result_list = run_util.parse_result(result_file, parse_file);

    run_count += 1
    
    return result_list

result_item_num = 4
result_list_all = [[0 for y in range(0,result_item_num)] for x in range(0,grid_size)]

for split_item in split_list:
    test_file = split_item
    train_file_list = filter(lambda x : x != split_item,split_list)

    train_file = test_file + '_train'
    os.system('rm -f %s' %train_file)
    for x in train_file_list:
        merge_cmd = 'cat %s' %x + ' >> %s' %train_file
        os.system(merge_cmd)

    cmd_data = dataset.get_cmd_data_by_file(train_file, test_file)
    result_list_one = run_one_data(exe_cmd + cmd_data)

    #raw_input('type to continue' )
    for k in range(0,grid_size):
        for m in range(0,result_item_num):
            result_list_all[k][m] += float(result_list_one[k][m])

    #delete the temp files
    os.system('rm -f %s' %train_file)
    os.system('rm -f %s' %train_file + '_cache')

#average the results
for k in range(0,grid_size):
        for m in range(0,result_item_num):
            result_list_all[k][m] /= fold_num
            result_list_all[k][m] = str(result_list_all[k][m]) 

#merge param_list and result_list
for k in range(0,grid_size):
    for item in param_list[k]:
        result_list_all[k].append(str(item))

final_item = min(result_list_all, key=lambda x:float(x[1])) #compare test error rate
final_str = ' '.join(str(x) for x in final_item)
print 'cross validation result: ' + final_str
final_file = dst_folder +'/cv_result.txt'
os.system('echo %s' %opt_name + ': %s' %final_str + ' >> %s' %final_file)
print 'cross validation result written to %s' %final_file
