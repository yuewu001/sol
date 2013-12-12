#!/usr/bin/env python
"""bach cross validation"""
import os
import sys
import dataset

#opt_list = ['STG','Ada-FOBOS','SSAROW', 'RDA','Ada-RDA', 'CW-RDA']
opt_list = ['STG','Ada-FOBOS', 'SSAROW','RDA','Ada-RDA', 'CW-RDA']
#opt_list = ['CW-RDA']

#ds_list = ['news', 'rcv1', 'url']
#ds_list = ['MNIST','news', 'rcv1','url']
#ds_list = ['webspam_trigram']
ds_list = ['MNIST','news','rcv1','url','aut','news20','gisette','physic','pcmac', 'real-sim','webspam_trigram']
#ds_list = ['news20']

fold_num = 5

eta_search = '1.0:2.0:1024.0'
delta_search = '0.03125:2:8'
r_search = delta_search

for dt in ds_list:
    #train model
    for opt in opt_list:
        print '----------------------------------------------'
        print 'Cross validation on %s' %dt + ' with %s' %opt
        print '----------------------------------------------'

        cmd = './CV.py {0} {1} {2} '.format(dt,opt,fold_num)

        if 'AROW' in opt or 'CW-RDA' in opt:
            cmd += ' {0} {1}'.format('-r',r_search)
        elif 'Ada' in opt:
            cmd += ' {0} {1}'.format('-delta',delta_search)
            cmd += ' {0} {1}'.format('-eta',eta_search)
        else:
            cmd += ' {0} {1}'.format('-eta',eta_search)
        print cmd
        os.system(cmd)

