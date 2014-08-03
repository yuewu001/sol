#!/usr/bin/env python
import os
import sys

import dataset
import util

import run_ofs
import run_liblinear
import run_fgm
import run_mRMR

#model list
model_list = ['SOFS','PET','PreSelOGD']
model_list = ['SOFS', 'PreSelOGD']

#dataset list
ds_list = ['relathe','pcmac','basehock','ccat','aut','real-sim']
ds_list = ['caltech']

#number of times to randomize a dataset for averaged results
rand_num = 10
#extra command sent to SOL
model_config = {
'cache':False,
'norm':False,
'bc_loss':'Hinge',
'mc_loss':'MaxScoreHinge',
'rand_num':rand_num,
'passes':10
}

#whether to use the default parameter settings of each algorithm, otherwise,
#parameters will obtained from get_model_param in dataset.py
is_default_param = False

#train model
def train_model(dataset):
    #random the file
    if rand_num > 1:
        rand_file = dataset.train_file + '_rand'  
    else:
        rand_file = dataset.train_file

    rand_file_cache = rand_file + '_cache'

    model_result_dict = {}
    for model in model_list:
        model_result_dict[model] = util.ResultItem()

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

            #output file
            result_file = '{0}/{1}_result_{2}.txt'.format(dst_folder,model, k)
            result_file = result_file.replace('/',os.sep)
            #clear the file if it already exists
            open(result_file,'w').close()

            if model == 'liblinear':
                result_once = run_liblinear.run(dataset, model_config, result_file)
            elif model == 'FGM':
                result_once = run_fgm.run(dataset, model_config, result_file)
            elif model == 'mRMR':
                param_config = ''
                #get parameters
                if is_default_param == False:
                    param_config = dataset.get_best_param('SGD')

                run_mRMR.run(dataset, model_config,param_config, result_file)
                continue
            else:
                param_config = ''
                #get parameters
                if is_default_param == False:
                    param_config = dataset.get_best_param(model)

                result_once = run_ofs.run(dataset,model, model_config,
                        param_config, result_file)


            model_result_dict[model].Add(result_once)

    #average the result
    if (rand_num > 1):
        for key,val in model_result_dict.iteritems():
            val.Divide(rand_num)

    return model_result_dict 

#train the model
for ds in ds_list:

    dt = dataset.dt_dict[ds]
    model_result_dict = train_model(dt)

    #write the result to file
    dst_folder = dt.name
    for key,val in model_result_dict.iteritems():
        result_file = dst_folder +'/%s' %key + '.txt'
        val.save_result(result_file)

