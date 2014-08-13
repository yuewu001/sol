#!/usr/bin/env python
import os
import sys

import dataset
import util

import run_ofs
import run_liblinear
import run_fgm
import run_mRMR
import run_bif

#model list
model_list = ['SOFS','PET','PreSelOGD-mRMR']

model_list = ['SOFS','PET','mRMR','BIF','PreSelOGD-mRMR','PreSelOGD-BIF']
model_list = ['BIF']

#dataset list
ds_list = ['relathe','pcmac','basehock','ccat','aut','real-sim']
ds_list = ['synthetic_10K', 'synthetic_20K', 'relathe','pcmac','basehock','ccat','aut','real-sim','news','rcv1','url'] 

#number of times to randomize a dataset for averaged results
rand_num = 1
#extra command sent to SOL
model_config = {
'cache':False,
'norm':True,
'bc_loss':'Hinge',
'mc_loss':'MaxScoreHinge',
'rand_num':rand_num,
'passes':1
}

#whether to use the default parameter settings of each algorithm, otherwise,
#parameters will obtained from get_model_param in dataset.py
is_default_param = False

#train model
def train_model(dataset):
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

            if model == 'liblinear':
                #output file
                result_file = '{0}/{1}_result_{2}.txt'.format(dst_folder,model, k)
                result_file = result_file.replace('/',os.sep)
                #clear the file if it already exists
                open(result_file,'w').close()

                result_once = run_liblinear.run(dataset, model_config, result_file)
            elif model == 'FGM':
                result_once = run_fgm.run(dataset, model_config, result_file)
            elif model == 'mRMR':
                run_mRMR.run(dataset, model_config)
                continue
            elif model == 'BIF':
                run_bif.run(dataset, model_config)
                continue
            else:
                #output file
                result_file = '{0}/{1}_result_{2}.txt'.format(dst_folder,model, k)
                result_file = result_file.replace('/',os.sep)
                #clear the file if it already exists
                open(result_file,'w').close()

                param_config = ''
                #get parameters
                if is_default_param == False:
                    param_config = dataset.get_best_param(model)

                result_once = run_ofs.run(dataset,model, model_config,
                        param_config, result_file)


            model_result_dict[model].Add(result_once)
        dataset.del_rand_file()

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

