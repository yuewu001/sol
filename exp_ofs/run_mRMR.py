#This script is to run experiment automatically to test the performance of the algorithm

import os
import time
import liblinear_util
import run_util
import exe_path
import l1_def
import re

#def parse_model_file(model_file):
model_file = 'model'
parse_file = 'model_parse'
print 'parse model file of mRMR%s\n' %model_file
c_feat = []
pattern = re.compile(r'(\S*)\s*')
is_begin = False
try:
    file_handler = open(model_file,'r')
    while True:
        line = file_handler.readline()
        line = line.strip()
        if is_begin == True and len(line) == 0:
            break
        if line == '*** mRMR features ***':
            line = file_handler.readline()
            is_begin = True
            continue
        if (is_begin == False):
            continue
        result_list = pattern.findall(line)
        c_feat.append(int(result_list[1]))
except IOError as e:
    print "I/O error ({0}): {1}".format(e.errno,e.strerror)
    sys.exit()
else:
    file_handler.close()
    c_feat.sort()
    print 'feature number %d' %(len(c_feat))
    print 'selected features:' 
    print c_feat
#write c_feat into file
try:
    file_handler = open(parse_file,'wb')

    for k in range(0,len(c_feat)):
        file_handler.write('%d\n' %c_feat[k])
except IOError as e:
    print "I/O error ({0}): {1}".format(e.errno,e.strerror)
    sys.exit()
else:
    file_handler.close()


def run_mRMR(train_file, test_file,ds, ori_result_file):
    result_all = []

    train_exe_name = exe_path.mRMR

    #make the result dir
    dst_folder = './%s' %ds
    run_util.create_dir(dst_folder)

    data_valid_dim = run_util.get_valid_dim(train_file)
    data_num = run_util.get_data_num(train_file)
    prev_cmd = train_exe_name + ' -s %d' %data_num + ' -v %d' %data_valid_dim + ' -t 0.5 -i %s' %train_file 

    if 'synthetic' in ds:
        bs_list = l1_def.get_lambda_list(dataset,'fgm')
    else:
        lambda_list = l1_def.get_lambda_list(ds,'fgm')

        bs_list = [] 
        b_num = len(lambda_list)
        for i in range(0,b_num):
            dim = int(data_valid_dim * (1 - lambda_list[i]))
            if dim > 0:
                bs_list.append(dim)

    for bs in bs_list:
        result_once = [0,0,0,0]
        model_file = dst_folder + '/mRMR_model%g' %bs

        result_file = dst_folder + '/' + ori_result_file + '_%d' %bs
        #clear the file if it already exists
        open(result_file,'w').close()

        cmd = prev_cmd + ' -n %d' %bs + ' > %s' %model_file
        print cmd
        start_time =time.time()
        os.system(cmd)
        end_time = time.time()

        #parse learning time
        train_time = (float)(end_time - start_time) 
        result_once[3] = train_time
        
        #predict
        print test_cmd
        os.system(test_cmd)
        result_once[2] = bs

        result_all.append(result_once)
    return result_all
