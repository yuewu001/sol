#!/usr/bin/env python
import os
import sys
import dataset
import run_util
import sol_shuffle
import run_mpol


#algorithm list
opt_list = ['SGD-FS','OFSGD','AROW-FS','STG','FOBOS','Ada-FOBOS','AROW-TG', 'RDA','Ada-RDA', 'AROW-DA']
opt_list = ['STG','FOBOS','Ada-FOBOS','AROW-TG', 'RDA','Ada-RDA', 'AROW-DA','AROW-FS']
#opt_list = ['STG','FOBOS','Ada-FOBOS','AROW-TG']
opt_list = ['FOBOS','Ada-FOBOS','AROW-TG']
#opt_list = ['AROW-FS','SGD-FS','OFSGD']
#opt_list = ['RDA','Ada-RDA','AROW-DA']
opt_list = ['SGD']

mp_list = ['none','all','reservior']
mp_list = ['margin']
#mp_list = ['all','false_predict']

#dataset list
ds_list = ['pcmac','aut','splice','a1a','a3a','a5a','gisette','w1a','w3a']

#number of times to randomize a dataset for averaged results
rand_num = 1
run_time = 10
#extra command sent to SOL
extra_cmd = ' -loss Hinge '

#whether need to cache the dataset for fast processing speed
is_cache = False
#whether to use the default parameter settings of each algorithm, otherwise,
#parameters will obtained from get_model_param in dataset.py
is_default_param = False

def add_to_dict(opt, mp_method, result_all, result_once):
    if opt not in result_all.keys(): #add to previous result
        result_all[opt] = {}
    if mp_method not in result_all[opt].keys(): #add to previous result
        result_all[opt][mp_method] = result_once
        result_all[opt][mp_method] = result_once
        rows = len(result_once)
        cols = len(result_once[0])
        for k in range(0,rows):
            for m in range(0,cols):
                result_all[opt][mp_method][k][m] = float(result_once[k][m])
    else:
        rows = len(result_all[opt][mp_method])
        cols = len(result_all[opt][mp_method][0])
        for k in range(0,rows):
            for m in range(0,cols):
                result_all[opt][mp_method][k][m] += float(result_once[k][m])
    return result_all

#train model
def train_model(path_list,dst_folder):
    train_file = path_list[0]
    test_file = path_list[1]

    result_all = {}
    #random the file
    if rand_num > 1:
        rand_file = train_file + '_rand'  
    else:
	    rand_file = train_file

    rand_file_cache = rand_file + '_cache'

    for k in range(0,rand_num):
        if rand_num > 1:
            print 'shuffle datset...'
            sol_shuffle.sol_shuffle(train_file, rand_file)

        cmd_data = dataset.get_cmd_data_by_file(rand_file, test_file, is_cache)
        dataset.analyze(rand_file);

        for opt in opt_list:
            for mp_method in mp_list:
                cmd_mp = ' -mpt %s ' %mp_method 
                for m in range(0,run_time):
                    print '-----------------------------------'
                    print ' Experiment on %s' %opt + ' Random %d' %k  + ' Multi-Pass %s' %mp_method + ' Round %d' %m
                    print '-----------------------------------'

                    result_file = dst_folder + '/%s' %opt + '_rand_%d' %k + '_mp_%s' %mp_method + '_round_%d' %m + '.txt'

                    cmd = cmd_data
                    cmd += cmd_mp
                    cmd += extra_cmd
                    if is_default_param == False:
                        cmd += dataset.get_model_param(ds, opt)

                    run_mpol.run_mpol(opt,result_file,ds, cmd)

                    print '\nparsing result...'
                    #write the result to file
                    parse_file = dst_folder +'/%s' %opt + '_%d' %k + '_%s' %mp_method + '_%d' %m  + '.txt'

                    result_once = run_util.parse_result(result_file, parse_file);
                    result_all = add_to_dict(opt,mp_method,result_all, result_once)

                    if mp_method == 'none':
                        break

        #remove previous file
        if rand_num > 1:
            os.system('rm -f %s' %rand_file_cache)
            os.system('rm -f %s' %rand_file)


    #average the result
    for opt in opt_list:
        for mp in mp_list:
            rows = len(result_all[opt][mp])
            cols = len(result_all[opt][mp][0])

            divid = rand_num
            if mp != 'none':
                divid *= run_time
            for k in range(0,rows):
                for m in range(0,cols):
                    result_all[opt][mp][k][m] /= divid

    return result_all 

for ds in ds_list:
    path_list = dataset.get_file_name(ds)
    dst_folder = ds
    dst_folder = ds
    #os.system("mkdir %s" %dst_folder)
    run_util.create_dir(dst_folder)

    result_all = train_model(path_list, dst_folder)

    for key,val in result_all.iteritems():
        #write the result to file
        for key2,val2 in val.iteritems():
            parse_file = dst_folder +'/%s' %key + '_%s' %key2 + '.txt'
            run_util.write_parse_result(val2,parse_file)
