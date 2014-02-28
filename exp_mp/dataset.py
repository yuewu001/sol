#!/usr/bin/env python
"""dataset list"""

import sys
import os
import platform
import exe_path

# windows
if platform.system() == 'Windows':
    rootDir = 'E:/users/v-wuyue/sol/data/'
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
    if dataset == 'news':
        train_file = 'news/news_train'
        test_file = 'news/news_test' 
    elif dataset == 'aut':
        train_file = 'aut/aut_train'
        test_file = 'aut/aut_test'
    elif dataset == 'pcmac':
        train_file = 'pcmac/pcmac_train'
        test_file = 'pcmac/pcmac_test'
    elif dataset == 'splice':
        train_file = 'splice/splice'
        test_file = 'splice/splice.t'
    elif dataset == 'a1a':
        train_file = 'a1a/a1a'
        test_file = 'a1a/a1a.t'
    elif dataset == 'a3a':
        train_file = 'a3a/a3a'
        test_file = 'a3a/a3a.t'
    elif dataset == 'a5a':
        train_file = 'a5a/a5a'
        test_file = 'a5a/a5a.t'
    elif dataset == 'a9a':
        train_file = 'a9a/a9a'
        test_file = 'a9a/a9a.t'
    elif dataset == 'gisette':
        train_file = 'gisette/gisette_scale'
        test_file = 'gisette/gisette_scale.t'
    elif dataset == 'w1a':
        train_file = 'w1a/w1a'
        test_file = 'w1a/w1a.t'
    elif dataset == 'w3a':
        train_file = 'w3a/w3a'
        test_file = 'w3a/w3a.t'
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
    news = {'SGD':{'-eta':64}}
    aut = {'SGD':{'-eta':32}}
    pcmac = {'SGD':{'-eta':8}}
    splice = {'SGD':{'-eta':1.0}}
    a1a = {'SGD':{'-eta':0.5}}
    a3a = {'SGD':{'-eta':0.25}}
    a5a = {'SGD':{'-eta':1.0}}
    gisette = {'SGD':{'-eta':0.25}}
    w1a = {'SGD':{'-eta':0.25}}
    w3a = {'SGD':{'-eta':0.25}}
    a9a = {'SGD':{'-eta':1.0}}
    news = {'SGD':{'-eta':64.0}}
    ds_opt_param = {'news':news,'aut':aut,'pcmac':pcmac,'splice':splice,'a1a':a1a,'a3a':a3a,'a5a':a5a,'a9a':a9a,'gisette':gisette,'w1a':w1a,'w3a':w3a} 
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
