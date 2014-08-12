#!/usr/bin/env python
"""convert libsvm format data to arff format"""

import sys
import re

def Usage():
    print 'libsvm2arff.py input_file output_file'

input_file = sys.argv[1]
output_file = sys.argv[2]

#count the number of attributes
index_feat_pattern = re.compile(r'(\d+:\d+\.?\d*)')
max_index = 0
class_num = 0
class_set = []
try:
    rfh = open(input_file,'r')
    while True:
        line = rfh.readline()
        if len(line) == 0:
            break
        match_result = index_feat_pattern.findall(line)
        if len(match_result) == 0:
            continue
        label = re.split(' |\t',line)[0].strip()
        label = label[1:] if label[0] == '+' else label
        label = 'class_' + label
        if label not in class_set: 
            class_set.append(label) 
        index_list = [int(item.split(':')[0]) for item in match_result ]
        tmp_max_index = max(index_list) 
        if max_index < tmp_max_index :
            max_index = tmp_max_index 

except IOError as e:
    print 'Error {0}: {1}'.format(e.errno, e.strerror)
    sys.exit()
else:
    rfh.close()

print 'classes: ', class_set
print 'max index %d' %max_index

print 'convert to arff format'
try:
    rfh = open(input_file,'r')
    wfh = open(output_file,'w')
    wfh.write('@relation libsvm2arff\n\n')
    for k in range(0,max_index):
        wfh.write('@attribute attribute_%d real\n' %(k + 1))
    wfh.write('@attribute class {' + ','.join(class_set) + '}\n\n')

    wfh.write('@data\n')
    while True:
        line = rfh.readline()
        if len(line) == 0:
            break
        label = re.split(' |\t',line)[0].strip()
        label = label[1:] if label[0] == '+' else label
        match_result = index_feat_pattern.findall(line)
        index_list = [int(item.split(':')[0]) - 1 for item in match_result ]
        feat_list = [item.split(':')[1] for item in match_result ]
        if len(match_result) == 0:
            continue
        feats = ['0' for k in range(0,max_index)]
        feat_num = len(index_list)
        for k in range(0,feat_num):
            feats[index_list[k]] = feat_list[k] 
        feats.append('class_%s\n'%label)
        wfh.write(','.join(feats))

except IOError as e:
    print 'Error {0}: {1}'.format(e.errno, e.strerror)
    sys.exit()
else:
    rfh.close()
    wfh.close()


