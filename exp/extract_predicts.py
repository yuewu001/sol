#! /usr/bin/env python
"""Extract the predicted results"""

import sys
import os
import ntpath

import dataset


def Usage():
    print 'extract_predicts.py dataset model model_size label'

def move_file(src, dst):
    with open(src, 'rb') as rfh:
        with open(dst, 'wb') as wfh:
            wfh.write(rfh.read())

def make_dir(path):
    if os.path.exists(path) == False:
        os.makedirs(path)


if len(sys.argv)  != 5:
    Usage()
    sys.exit()

dt = dataset.dt_dict[sys.argv[1]]
model = sys.argv[2]
model_size = sys.argv[3]
label = sys.argv[4]

predict_file = '%s/%s/predict_%s.txt' %(dt.name, model, model_size)

test_path_file = '%s%s/testPaths.txt' %(dt.root_dir, dt.name)
paths = []
with open(test_path_file,'r') as fh:
    while True:
        path = fh.readline().strip()
        if len(path) == 0:
            break
        paths.append(path)

img_id = 0
dst_true_folder = '%s/%s/%s/true/' %(dt.name,model,label)
dst_false_folder = '%s/%s/%s/false/' %(dt.name,model,label)
make_dir(dst_true_folder)
make_dir(dst_false_folder)

with open(predict_file, 'r') as fh:
    while True:
        result = filter(None,fh.readline().strip().split('\t'))
        img_id += 1
        if len(result) == 0:
            break
        elif len(result) != 2:
            print result
            print 'incorrect file format'
            break
        if result[1] != label:
            continue
        img_path = paths[img_id - 1]
        img_name = ntpath.basename(img_path)
        if result[0] != result[1]:
            src_path = '%s%s/images/%s' %(dt.root_dir, dt.name,img_path)
            dst_path = dst_false_folder +  img_name
            move_file(src_path, dst_path)
        else:
            src_path = '%s%s/images/%s' %(dt.root_dir, dt.name,img_path)
            dst_path = dst_true_folder +  img_name
            move_file(src_path, dst_path)


