#!/usr/bin/env python
# AUTHOR:   Yue Wu (yuewu@outlook.com)
# FILE:     run_sol.py
# ROLE:     TODO (some explanation)
# CREATED:  2015-02-08 00:41:09
# MODIFIED: 2015-02-08 20:47:38
#This script is to run experiment automatically to test the performance of the algorithm

import sys
import os

import util

import run_mRMR
import run_bif

#sol executable
if util.get_platform() == 'Windows':
    sol_exe = r'..\install\bin\SOL.exe'
else:
    sol_exe = '../install/bin/SOL'

def get_cmd_prefix(dt, model, config):
    dt_cmd = dt.get_train_cmd(config['rand_num'],config['cache'])
    if dt.class_num > 2:
        if model == 'CW_TG' or model == 'CW_RDA':
            loss_cmd = ' -cn %d -loss MaxScoreSquaredHinge ' %(dt.class_num)
        else:
            loss_cmd = ' -cn {0} -loss {1} '.format(dt.class_num, config['mc_loss'])
    else:
        if model == 'CW_TG' or model == 'CW_RDA':
            loss_cmd = ' -cn 2 -loss SquaredHinge '
        else:
            loss_cmd = ' -cn 2 -loss {0} '.format(config['bc_loss'])

    norm_cmd = ' -norm ' if config['norm'] == True else '' 

    #get parameters
    if config['is_default_param'] == False:
        param_config = dt.get_best_param(model)

    cmd_prefix  = sol_exe + dt_cmd + loss_cmd + norm_cmd  + ' -m %s ' %model + param_config

    if model == 'RDA':
        cmd_prefix += ' -grou 0 '

    if 'passes' in config:
        cmd_prefix += ' -passes %d ' %config['passes']

    return cmd_prefix



#run the online feature selection experiment
#@param dataset: DataSet instance
#@param model: model to train 
#@param config: cnfiguration to train the model
#@param output_file: output file to save the results
def run_all(dt, model, config, output_file):
    l1_list = dt.get_lambda_list(model)

    cmd_prefix = get_cmd_prefix(dt, model,config)
    #evaluate the result
    cmd_postfix = ' >> %s' %output_file

    for l1 in l1_list:
        cmd = cmd_prefix + ' -l1 {0}'.format(l1)
        predict_file   = dt.name + '/%s/predict_%g.txt' %(model,l1)
        cmd += ' -op %s ' %predict_file
        cmd += cmd_postfix
        cmd = cmd.replace('/',os.sep)
        print cmd
        os.system(cmd)

    #parse the result
    result = util.ResultItem()
    print output_file
    result.parse_ofs_result(output_file)

    print '\nTraining Result: '
    result.Display()
    print '\n'

    return result

def run_once(cmd_prefix, l1, output_file):
    print 'get test error rate when l1 == {0}'.format(l1)
    cmd = cmd_prefix + ' -l1 {0}'.format(l1)
    cmd += ' > %s' %output_file
    cmd = cmd.replace('/',os.sep)
    print cmd
    os.system(cmd)

    result = util.ResultItem()
    result.parse_ofs_result(output_file)
    print 'train_error: {0}\ttest_error: {1}'.format(result.train_error[0], result.test_error[0])

    return result

#run the online feature selection experiment
#@param dataset: DataSet instance
#@param model: model to train 
#@param config: cnfiguration to train the model
#@param output_file: output file to save the results
def search_best_l1(dt, model, config, output_file):
    cmd_prefix = get_cmd_prefix(dt, model,config)

    tolerance = config['tolerance']
    #search the [left, right] range where l1 will be searched
    l1_left = 0
    result_left = run_once(cmd_prefix,l1_left, output_file)
    worst_test_error_rate = 1.0 - (1.0 - result_left.test_error[0]) * (1 - tolerance)

    l1_right = 0.01
    result_right = run_once(cmd_prefix,l1_right, output_file)

    while result_right.test_error[0] <= worst_test_error_rate:
        l1_right *= 2
        result_right = run_once(cmd_prefix,l1_right, output_file)

    #search the best l1
    stop_l1_step = 1e-7
    stop_test_error_rate = 0.01
    iter_num = 0
    while l1_right - l1_left > stop_l1_step and \
        abs(result_right.test_error[0] - result_left.test_error[0]) > stop_test_error_rate :
        iter_num += 1
        print 'iteration: %d' %iter_num

        l1 = (l1_left + l1_right) / 2.0
        result = run_once(cmd_prefix, l1, output_file)
        ter = result.test_error[0]
        if ter > worst_test_error_rate:
            result_right = result
            l1_right = l1
        else:
            result_left = result
            l1_left = l1

    print '\nTraining Result: '
    print 'best l1: ', l1
    result.Display()
    print '\n'

    return result
