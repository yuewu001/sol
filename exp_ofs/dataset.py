#!/usr/bin/env python
"""dataset list"""

import sys
import os
import platform
import exe_path

# windows
if platform.system() == 'Windows':
    rootDir = 'D:/v-wuyue/data/'
elif platform.system() == 'Linux':
    rootDir = '/root/v-yuewu/SOL/data/'
else:
    print 'system type is not supported:'
    sys.exit()

#analyze dataset
def analyze(file_name):
    info_name = file_name + '_info.txt'
    if os.path.exists(info_name) == False:
        print 'analyze %s' %file_name
        cmd = exe_path.analysis_exe_name + ' %s' %file_name +' >> %s' %info_name
        print cmd
        os.system(cmd)

def get_file_name(dataset, task = 'train'):
    if dataset == 'aut':
        train_file = 'aut/aut_train'
        test_file = 'aut/aut_test'
    elif dataset == 'pcmac':
        train_file = 'pcmac/pcmac_train'
        test_file = 'pcmac/pcmac_test'
    elif dataset == 'basehock':
        train_file = 'basehock/basehock_train'
        test_file = 'basehock/basehock_test' 
    elif dataset == 'relathe':
        train_file = 'relathe/relathe_train'
        test_file = 'relathe/relathe_test' 
    elif dataset == 'ccat':
        train_file = 'ccat/ccat_train' 
        test_file = 'ccat/ccat_test'
    elif dataset == 'gisette':
        train_file = 'gisette/gisette_scale' 
        test_file = 'gisette/gisette_scale.t'
    elif dataset == 'real-sim':
        train_file = 'real-sim/real-sim_train' 
        test_file = 'real-sim/real-sim_test'
    elif dataset == 'news':
        train_file = 'news/news_train'
        test_file = 'news/news_test' 
    elif dataset == 'rcv1':
        train_file = 'rcv1/rcv1_train'
        test_file = 'rcv1/rcv1_test' 
    elif dataset == 'url':
        train_file = 'url_combined/url_train'
        test_file = 'url_combined/url_test' 
    elif dataset == 'synthetic_100_10K_100K':
        train_file = 'synthetic_ofs/100_10K_100K/synthetic_train'
        test_file = 'synthetic_ofs/100_10K_100K/synthetic_test'
    elif dataset == 'synthetic_200_20K_100K':
        train_file = 'synthetic_ofs/200_20K_100K/synthetic_train'
        test_file = 'synthetic_ofs/200_20K_100K/synthetic_test'
    elif dataset == 'synthetic_200_1M_1M':
        train_file = 'synthetic_ofs/200_1M_1M/synthetic_train'
        test_file = 'synthetic_ofs/200_1M_1M/synthetic_test'
    else:
        print 'unrecoginized dataset'
        sys.exit()

    train_file = rootDir + train_file
    test_file = rootDir + test_file
    if os.sep != '/':
        train_file = train_file.replace('/', os.sep)
        test_file = test_file.replace('/', os.sep)

    path_list = ['','']
    path_list[0] = train_file
    path_list[1] = test_file

    return path_list

def split_dataset(dataset, fold_num):
    path_list = get_file_name(dataset,'cv')
    train_file = path_list[0]

    # count number of lines
    count_cmd = 'wc -l %s' %train_file
    count_handler = os.popen(count_cmd)
    line_num = int(count_handler.read().split()[0])
    count_handler.close()

    #split the train_data into fold_num pieces
    split_line_num = int(line_num / fold_num)

    #catch all the sub files, note that fold-num must less 26
    split_list = []
    for k in range(0,fold_num):
        file_name = train_file + '_cva' + chr(ord('a') + k)
        os.system('rm -f %s' %file_name)
        split_list.append(file_name)

    split_cmd = 'split -l {0} {1} {2}_cv'\
            .format(split_line_num,train_file, train_file) 

    os.system(split_cmd)
    return split_list

def get_cv_data_list(dataset, fold_num):
    path_list = get_file_name(dataset,'cv')
    train_file = path_list[0]

    #catch all the sub files, note that fold-num must less 26
    split_list = []
    for k in range(0,fold_num):
        file_name = train_file + '_cva' + chr(ord('a') + k)
        split_list.append(file_name)

    return split_list

def get_cmd_data_by_file(train_file, test_file, is_cache = True):
    cmd_data = ' -i %s' %train_file 
    cmd_data += ' -t %s' %test_file
    if is_cache == True:
        cache_train_file = train_file + '_cache'
        cmd_data += ' -c %s' %cache_train_file

        cache_test_file = test_file + '_cache'
        cmd_data += ' -tc %s' %cache_test_file

    return cmd_data

def get_cmd_data(dataset):
    path_list = get_file_name(dataset)
    train_file = path_list[0]
    test_file = path_list[0]

    analyze(train_file)
    analyze(test_file)

    return get_cmd_data_by_file(train_file, test_file)

#parameters for each dataset, obtained by cross-validation in general
def get_model_param(ds, opt):
    aut = {'AROW-FS':{'-r':0.5},'SGD-FS':{'-eta':32},'OFSGD':{'-eta':0.25,'-delta':0.0003125}}
    pcmac = {'AROW-FS':{'-r':1.0},'SGD-FS':{'-eta':8},'OFSGD':{'-eta':1.0,'-delta':0.000625}}
    basehock = {'AROW-FS':{'-r':0.5},'SGD-FS':{'-eta':8},'OFSGD':{'-eta':0.5,'-delta':0.0003125}}
    relathe = {'AROW-FS':{'-r':0.25},'SGD-FS':{'-eta':32},'OFSGD':{'-eta':4.0,'-delta':0.0003125}}
    ccat = {'AROW-FS':{'-r':1.0},'SGD-FS':{'-eta':256},'OFSGD':{'-eta':0.25,'-delta':0.0003125}}
    gisette = {'AROW-FS':{'-r':0.25},'SGD-FS':{'-eta':128},'OFSGD':{'-eta':0.125,'-delta':0.0003125}}
    real_sim = {'AROW-FS':{'-r':0.5},'SGD-FS':{'-eta':16},'OFSGD':{'-eta':0.25,'-delta':0.0003125}}
    news = {'AROW-FS':{'-r':0.25},'SGD-FS':{'-eta':64},'OFSGD':{'-eta':0.25,'-delta':0.0003125}}
    rcv1 = {'AROW-FS':{'-r':2},'SGD-FS':{'-eta':32},'OFSGD':{'-eta':0.25,'-delta':0.0003125}}
    url = {'AROW-FS':{'-r':0.25},'SGD-FS':{'-eta':256},'OFSGD':{'-eta':0.25,'-delta':0.0003125}}
    synthetic_100_10K_100K = {'AROW-FS':{'-r':128},'SGD-FS':{'-eta':8},'OFSGD':{'-eta':0.0625,'-delta':0.0003125}}
    synthetic_200_20K_100K = {'AROW-FS':{'-r':64},'SGD-FS':{'-eta':8},'OFSGD':{'-eta':0.03125,'-delta':0.0003125}}
    synthetic_200_1M_1M = {'AROW-FS':{'-r':64},'SGD-FS':{'-eta':8},'OFSGD':{'-eta':0.03125,'-delta':0.0003125}}
    ds_opt_param = {'aut':aut,'pcmac':pcmac,'basehock':basehock,'relathe':relathe,'ccat':ccat,'gisette':gisette,
            'real-sim':real_sim,'synthetic_100_10K_100K':synthetic_200_20K_100K,'synthetic_100_10K_100K':synthetic_200_20K_100K,'synthetic_100_1M_1M':synthetic_200_1M_1M,'rcv1':rcv1,'news':news,'url':url}
    
    cmd = ''
    if ds in ds_opt_param.keys():
        if opt in ds_opt_param[ds].keys():
            for key,val in ds_opt_param[ds][opt].iteritems():
                cmd += ' {0} {1} '.format(key, val)
        else:
            print opt + ' not unrecognized!'
    else:
        print ds + ' not unrecognized!'
    return cmd
