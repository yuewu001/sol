#!/usr/bin/env python
import os
import sys

#opt_list = ['STG','Ada-FOBOS','SSAROW', 'RDA','Ada-RDA', 'CW-RDA']
opt_list = ['STG','Ada-FOBOS', 'SSAROW','RDA','Ada-RDA', 'CW-RDA','ASAROW']
#opt_list = ['ASAROW']

#dataset_list = ['news', 'rcv1', 'url']
dataset_list = ['MNIST','news', 'rcv1','url','webspam_trigram']
#dataset_list = ['webspam_trigram']

#rootDir = 'D:/Coding/SOL/data/'
rootDir = '/home/matthew/work/Data/'

for dataset in dataset_list:
    cmd_data = ' -loss Hinge -norm '
    test_file = ''
    if dataset == 'rcv1':
        train_file = 'rcv1/rcv1.train' 
        test_file = 'rcv1/rcv1.test'
    elif dataset == 'news':
        train_file = 'news/news_train'
        test_file = 'news/news_test' 
    elif dataset == 'real-sim':
        train_file = 'real-sim/real-sim_train'
        test_file = 'real-sim/real-sim_test' 
    elif dataset == 'epsilon':
        train_file = 'epsilon/epsilon_normalized'
        test_file = 'epsilon/epsilon_normalized.t'
    elif dataset == 'url':
        train_file = 'url_combined/url_train'
        test_file  = 'url_combined/url_test'
    elif dataset == 'webspam':
        train_file = 'webspam/webspam_unigram_train'
        test_file  = 'webspam/webspam_unigram_test'
    elif dataset == 'webspam_trigram':
        train_file = 'webspam_trigram/webspam_trigram_train'
        test_file  = 'webspam_trigram/webspam_trigram_test'
    elif dataset =='MNIST':
        train_file = 'MNIST/train67'
        test_file  = 'MNIST/test67'
    else:
        print 'unrecoginized dataset'
        sys.exit()
    
    train_file = rootDir + train_file

    if os.sep != '/':
        train_file = train_file.replace('/', os.sep)
        test_file = test_file.replace('/', os.sep)

    cmd_data += ' -i %s' %train_file 
    cache_train_file = train_file + '_cache'
    cmd_data += ' -c %s' %cache_train_file

    if len(test_file) > 0:
        test_file = rootDir+ test_file
        cache_test_file = test_file + '_cache'
        cmd_data += ' -t %s' %test_file + ' -tc %s' %cache_test_file
    else:
        cache_test_file = ''
    
    dst_folder = dataset
    os.system("mkdir %s" %dst_folder)
    
    #analyze dataset
    dataset_info_file = train_file + '_info.txt'
    if os.path.exists(dataset_info_file) == False:
        print 'analyze dataset'
        cmd = '..' + os.sep + 'analysis %s' %train_file +' >> %s' %dataset_info_file
        print cmd
        os.system(cmd)
    
    #train model
    for opt in opt_list:
        print '-----------------------------------'
        print 'Experiment on %s' %opt
        print '-----------------------------------'
        cmd = 'python run_experiment.py %s' %opt  + ' %s' %dst_folder + ' %s' %dataset
        cmd += cmd_data

        if dataset == 'news':
            if opt == 'Ada-FOBOS':
                cmd += ' -eta 16 -delta 1 '
            elif opt == 'Ada-RDA':
                cmd += ' -eta 16 -delta 2 '
            elif opt == 'STG':
                cmd += ' -eta 32 '
            elif opt == 'RDA':
                cmd += ' -eta 32'
	    elif opt == 'SSAROW' or opt == 'CW-RDA' or opt == 'ASAROW':
            	cmd += ' -r 0.125 '
            else:
                print 'unrecognized %s' %opt
                sys.exit()
	elif dataset == 'MNIST':
            if opt == 'SSAROW' or opt == 'ASAROW':
                cmd += ' -r 1 '
            elif opt == 'CW-RDA':
                cmd += '-r 2'
            elif opt == 'Ada-FOBOS':
                cmd += ' -eta 8 -delta 8 '
            elif opt == 'Ada-RDA':
                cmd += ' -eta 4 -delta 8 '
            elif opt == 'STG':
                cmd += ' -eta 4 '
            elif opt == 'RDA':
                cmd += ' -eta 8 '
            else:
                print 'unrecognized %s' %opt
                sys.exit()

        elif dataset == 'rcv1':
            if opt == 'SSAROW' or opt == 'ASAROW':
                cmd += ' -r 1 '
            elif opt == 'CW-RDA':
                cmd += '-r 2'
            elif opt == 'Ada-FOBOS':
                cmd += ' -eta 4 -delta 8 '
            elif opt == 'Ada-RDA':
                cmd += ' -eta 4 -delta 8 '
            elif opt == 'STG':
                cmd += ' -eta 32 '
            elif opt == 'RDA':
                cmd += ' -eta 64'
            else:
                print 'unrecognized %s' %opt
                sys.exit()
        elif dataset == 'url':
            if opt == 'SSAROW' or opt == 'ASAROW':
                cmd += ' -r 1 '
            elif opt == 'CW-RDA':
                cmd += '-r 2'
            elif opt == 'Ada-FOBOS':
                cmd += ' -eta 4 -delta 8 '
            elif opt == 'Ada-RDA':
                cmd += ' -eta 4 -delta 8 '
            elif opt == 'STG':
                cmd += ' -eta 32 '
            elif opt == 'RDA':
                cmd += ' -eta 64'
            else:
                print 'unrecognized %s' %opt
                sys.exit()
        elif dataset == 'real-sim':
            if opt == 'SSAROW' or opt == 'ASAROW':
                cmd += ' -r 4 '
            elif opt == 'CW-RDA':
                cmd += '-r 4'
            elif opt == 'Ada-FOBOS':
                cmd += ' -eta 4 -delta 8 '
            elif opt == 'Ada-RDA':
                cmd += ' -eta 4 -delta 8 '
            elif opt == 'STG':
                cmd += ' -eta 1 '
            elif opt == 'RDA':
                cmd += ' -eta 1'
            else:
                print 'unrecognized %s' %opt
                sys.exit()
        elif dataset == 'webspam': 
            if opt == 'SSAROW' or opt == 'ASAROW':
                cmd += ' -r 0.125 '
            elif opt == 'CW-RDA':
                cmd += '-r 0.125 '
            elif opt == 'Ada-FOBOS':
                cmd += ' -eta 8 -delta 0.125 '
            elif opt == 'Ada-RDA':
                cmd += ' -eta 8 -delta 0.5 '
            elif opt == 'STG':
                cmd += ' -eta 64 '
            elif opt == 'RDA':
                cmd += ' -eta 32 '
            else:
                print 'unrecognized %s' %opt
                sys.exit()
	elif dataset == 'webspam_trigram': 
            if opt == 'SSAROW' or opt == 'ASAROW':
                cmd += ' -r 0.125 '
            elif opt == 'CW-RDA':
                cmd += '-r 0.125 '
            elif opt == 'Ada-FOBOS':
                cmd += ' -eta 16 -delta 0.125 '
            elif opt == 'Ada-RDA':
                cmd += ' -eta 16 -delta 0.125 '
            elif opt == 'STG':
                cmd += ' -eta 128 '
            elif opt == 'RDA':
                cmd += ' -eta 128 '
            else:
                print 'unrecognized %s' %opt
                sys.exit()

        os.system(cmd)

    #sys.exit()
    opt_list_file = '%s' %dst_folder + os.sep + 'opt_list.txt' 
    #clear the file if it already exists
    open(opt_list_file,'w').close()

    try:
        file_handle = open(opt_list_file,'w')
        for opt in opt_list:
            file_handle.write(opt + '.txt\n')
    except IOError as e:
        print "I/O error ({0}): {1}".format(e.errno,e.strerror)
        sys.exit()
    else:
        file_handle.close()
