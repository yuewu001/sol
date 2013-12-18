import os
import sys
import run_experiment

#opt_list = ['liblinear']

ds_list = ['news','news20']
rand_num = 10
extra_cmd = ' -loss Hinge -norm '

is_cache = True
if is_cache == True:
    opt_list = ['SGD','vw']
else:
    opt_list = ['SGD','vw','liblinear']


dst_folder = '%s' %dataset + '_sgd_comp'

#train model
if is_cache == False:
    for opt in opt_list:
        if opt == 'vw':
            cmd = 'python run_vw.py %s' %dst_folder + ' %s' %train_file 
            cmd += ' nocache'  + ' %s' %test_file + ' nocache' 
            cmd += ' no_l1 no_cache'
            print cmd
            os.system(cmd)
        elif opt == 'liblinear':
            cmd = 'python run_liblinear.py %s' %dst_folder + ' ll_model'
            cmd += ' %s' %train_file + ' %s' %test_file
            print cmd
            os.system(cmd)
        elif opt == 'SGD':
            cmd = 'python run_experiment.py %s' %opt  + ' %s' %dst_folder
            cmd += cmd_data
            cmd += ' no_l1'
            print cmd
            os.system(cmd)
else:
    for opt in opt_list:
        if opt == 'vw':
            cmd = 'python run_vw.py %s' %dst_folder + ' %s' %train_file 
            cmd += ' %s' %cache_train_file + ' %s' %test_file + ' %s' %cache_test_file 
            cmd += ' no_l1'
            print cmd
            os.system(cmd)
        elif opt == 'SGD':
