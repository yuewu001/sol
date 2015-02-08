#!/usr/bin/env python
# AUTHOR:   Yue Wu (yuewu@outlook.com)
# FILE:     demo_util.py
# ROLE:     shared parts for demo
# CREATED:  2015-02-08 16:09:41
# MODIFIED: 2015-02-08 22:24:12

import os
import sys

import dataset
import util

import run_ofs
import run_sol
import run_liblinear
import run_fgm
import run_mRMR
import run_bif

#train feature selection
def train_fs(dataset, model, model_config, result_file):
    if model == 'liblinear':
        result_once = run_liblinear.run(dataset, model_config, result_file)
    elif model == 'FGM':
        result_once = run_fgm.run(dataset, model_config, result_file)
    elif model == 'mRMR':
        run_mRMR.run(dataset, model_config)
    elif model == 'BIF':
        run_bif.run(dataset, model_config)
    else:
        param_config = ''
        #get parameters
        if model_config['is_default_param'] == False:
            param_config = dataset.get_best_param(model)

        result_once = run_ofs.run(dataset,model, model_config, param_config, result_file)
    return result_once

#train sparse online learning
def train_sol(dataset, model, model_config, result_file):
    param_config = ''

    if model_config['sol_type'] == 'run_all':
        result_once = run_sol.run_all(dataset,model, model_config, result_file)
    elif model_config['sol_type'] == 'search_l1':
        result_once = run_sol.search_best_l1(dataset,model, model_config,result_file)
    else:
        print 'unrecognized sol_type: ', model_config['sol_type']
        sys.exit()
    return result_once


#train model
def train_model(dataset, model_list, model_config):
    model_result_dict = {}
    for model in model_list:
        model_result_dict[model] = util.ResultItem()

    model_type = model_config['type'].lower()
    rand_num = model_config['rand_num']

    for k in range(0,rand_num):
        if rand_num > 1:
            print 'shuffle datset...'
            dataset.shuffle_file()

        for model in model_list:
            print '-----------------------------------'
            print ' Experiment on %s' %model + ' Random %d' %k 
            print '-----------------------------------'

            #create destination folder
            dst_folder = dataset.name + '/%s' %model
            if os.path.exists(dst_folder) == False:
                os.makedirs(dst_folder)

            if model_type == 'fs':
                #output file
                result_file = '{0}/{1}_rand_{2}_type_{3}.txt'.format(dst_folder,model, k, model_type)
            elif model_type == 'sol' and model_config['sol_type'] == 'run_all':
                #output file
                result_file = '{0}/{1}_rand_{2}_type_{3}_soltype_{4}.txt'.format(dst_folder,model, k,model_type, model_config['sol_type'])
            elif model_type == 'sol' and model_config['sol_type'] == 'search_l1':
                result_file = '{0}/{1}_rand_{2}_type_{3}_soltype_{4}_tolerance_{5}.txt'.format(dst_folder,model, k,model_type, model_config['sol_type'], model_config['tolerance'])
            else:
                print 'unregnized model type: ', model['type']
                sys.exit()

            result_file = result_file.replace('/',os.sep)
            #clear the file if it already exists
            open(result_file,'w').close()

            if model_type == 'fs':
                result_once = train_fs(dataset, model, model_config, result_file)
            elif model_type == 'sol':
                result_once = train_sol(dataset, model, model_config, result_file)

            model_result_dict[model].Add(result_once)
        dataset.del_rand_file()

    #average the result
    if (rand_num > 1):
        for key,val in model_result_dict.iteritems():
            val.Divide(rand_num)

    return model_result_dict 

def demo(ds_list, model_list, model_config):
#train the model
    for ds in ds_list:
        dt = dataset.dt_dict[ds]
        model_result_dict = train_model(dt, model_list, model_config)

        #write the result to file
        dst_folder = dt.name
        for key,val in model_result_dict.iteritems():
            result_file = dst_folder +'/%s' %key + '.txt'
            val.save_result(result_file)
