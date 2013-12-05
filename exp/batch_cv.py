#!/usr/bin/env python
"""bach cross validation"""
import os
import sys
import dataset

#opt_list = ['STG','Ada-FOBOS','SSAROW', 'RDA','Ada-RDA', 'CW-RDA']
opt_list = ['STG','Ada-FOBOS', 'SSAROW','RDA','Ada-RDA', 'CW-RDA']
#opt_list = ['SSAROW','STG']

#ds_list = ['news', 'rcv1', 'url']
ds_list = ['MNIST','news', 'rcv1','url']
#ds_list = ['news']

fold_num = 5

eta_search = '1.0:2.0:128.0'
delta_search = '0.125:2:16'
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

