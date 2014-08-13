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

    #get the dimension of the data
    data_dim = dataset.dim

    if '+' in model:
        [model,pre_model] = filter(None,model.split('+'))
    else:
        pre_model = ''

    if pre_model == 'mRMR':
        sel_feat_num_list = dataset.mrmr_l0_list
    else:
        sel_feat_num_list = dataset.l0_list

    #evaluate the result
    cmd_postfix = ' >> %s' %output_file

    dt_cmd = dataset.get_train_cmd(config['rand_num'],config['cache'])
    if dataset.class_num > 2:
        if model == 'SOFS':
            loss_cmd = ' -cn %d -loss MaxScoreSquaredHinge ' %(dataset.class_num)
        else:
            loss_cmd = ' -cn {0} -loss {1} '.format(dataset.class_num, config['mc_loss'])
    else:
        if model == 'SOFS':
            loss_cmd = ' -cn 2 -loss SquaredHinge '
        else:
            loss_cmd = ' -cn 2 -loss {0} '.format(config['bc_loss'])

    norm_cmd = ' -norm ' if config['norm'] == True else '' 

    cmd_prefix  = ofs_exe + dt_cmd + loss_cmd + norm_cmd  + ' -m %s ' %model + param_config

    if 'passes' in config:
        cmd_prefix += ' -passes %d ' %config['passes']

    for sel_num in sel_feat_num_list:
        cmd = cmd_prefix + ' -k %d' %sel_num 
        if len(pre_model) != 0:
            model_file = dataset.name + '/%s/model_%d' %(pre_model,sel_num)
            cmd += ' -im %s ' %(model_file)
        #predict file
        predict_file   = dataset.name + '/%s+%s/predict_%g.txt' %(model,pre_model, sel_num)
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

    if len(pre_model) != 0:
        for k in range(0,len(sel_feat_num_list)):
            model_file = dataset.name + '/%s/model_%d' %(pre_model,sel_feat_num_list[k])
            result.train_time[k] += util.parse_train_time(model_file)

    print '\nTraining Result: '
    result.Display()
    print '\n'

    return result



