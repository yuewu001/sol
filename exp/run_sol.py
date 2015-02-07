#!/usr/bin/env python
# AUTHOR:   Yue Wu (yuewu@outlook.com)
# FILE:     run_sol.py
# ROLE:     TODO (some explanation)
# CREATED:  2015-02-08 00:41:09
# MODIFIED: 2015-02-08 01:01:30
#This script is to run experiment automatically to test the performance of the algorithm

import sys
import os

import util

import run_mRMR
import run_bif

#run the online feature selection experiment
#@param dataset: DataSet instance
#@param model: model to train 
#@param param_config: parameter configuration
#@param config: cnfiguration to train the model
#@param output_file: output file to save the results
def run(dataset, model, config, param_config, output_file):
    #ofs executable
    if util.get_platform() == 'Windows':
        ofs_exe = r'..\install\bin\SOL.exe'
    else:
        ofs_exe = '../install/bin/SOL'


    l1_list = dataset.get_lambda_list(model)
    print l1_list

    #evaluate the result
    cmd_postfix = ' >> %s' %output_file

    dt_cmd = dataset.get_train_cmd(config['rand_num'],config['cache'])
    if dataset.class_num > 2:
        if model == 'CW_TG' or model == 'CW_RDA':
            loss_cmd = ' -cn %d -loss MaxScoreSquaredHinge ' %(dataset.class_num)
        else:
            loss_cmd = ' -cn {0} -loss {1} '.format(dataset.class_num, config['mc_loss'])
    else:
        if model == 'CW_TG' or model == 'CW_RDA':
            loss_cmd = ' -cn 2 -loss SquaredHinge '
        else:
            loss_cmd = ' -cn 2 -loss {0} '.format(config['bc_loss'])

    norm_cmd = ' -norm ' if config['norm'] == True else '' 

    cmd_prefix  = ofs_exe + dt_cmd + loss_cmd + norm_cmd  + ' -m %s ' %model + param_config

    if 'passes' in config:
        cmd_prefix += ' -passes %d ' %config['passes']

    for l1 in l1_list:
        cmd = cmd_prefix + ' -l1 {0}'.format(l1)
        predict_file   = dataset.name + '/%s/predict_%g.txt' %(model,l1)
        cmd += ' -op %s ' %predict_file
        cmd += cmd_postfix
        cmd = cmd.replace('/',os.sep)
        print cmd
        os.system(cmd)

    #parse the result
    result = util.ResultItem()
    print output_file
    result.parse_ofs_result(output_file)
    result.Display()

    print '\nTraining Result: '
    result.Display()
    print '\n'

    return result

