#!/usr/bin/env python
import os
import sys
import dataset

#opt_list = ['STG','Ada-FOBOS','SSAROW', 'RDA','Ada-RDA', 'CW-RDA']
#opt_list = ['STG','Ada-FOBOS', 'SSAROW','RDA','Ada-RDA', 'CW-RDA','ASAROW']
opt_list = ['ASAROW']

#ds_list = ['news', 'rcv1', 'url']
#ds_list = ['MNIST','news', 'rcv1','url','webspam_trigram']
ds_list = ['rcv1']

extra_cmd = ' -loss Hinge -norm '
for ds in ds_list:
    cmd_data = dataset.get_cmd_data(ds)
        
    dst_folder = ds
    os.system("mkdir %s" %dst_folder)
    
    #train model
    for opt in opt_list:
        print '-----------------------------------'
        print 'Experiment on %s' %opt
        print '-----------------------------------'
        cmd = 'python run_experiment.py %s' %opt  + ' %s' %dst_folder + ' %s' %ds
        cmd += cmd_data
        cmd += extra_cmd

        if ds == 'news':
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
	elif ds == 'MNIST':
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

        elif ds == 'rcv1':
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
        elif ds == 'url':
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
        elif ds == 'real-sim':
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
        elif ds == 'webspam': 
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
	elif ds == 'webspam_trigram': 
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

    sys.exit()
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
