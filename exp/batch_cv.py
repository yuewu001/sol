#!/usr/bin/env python
"""bach cross validation"""
import os

opt_list = ['STG','Ada-FOBOS','AROW-TG', 'RDA','Ada-RDA', 'AROW-DA']

ds_list = ['a9a','aut','MNIST','pcmac','physic','news']

fold_num = 5

eta_search = '0.5:2.0:512'
delta_search = '0.03125:2:32'
r_search = delta_search

for dt in ds_list:
    #train model
    for opt in opt_list:
        print '----------------------------------------------'
        print 'Cross validation on %s' %dt + ' with %s' %opt
        print '----------------------------------------------'

        cmd = 'CV.py {0} {1} {2} '.format(dt,opt,fold_num)

        if 'AROW' in opt or 'CW-RDA' in opt:
            cmd += ' {0} {1}'.format('-r',r_search)
        elif 'Ada' in opt:
            cmd += ' {0} {1}'.format('-delta',delta_search)
            cmd += ' {0} {1}'.format('-eta',eta_search)
        else:
            cmd += ' {0} {1}'.format('-eta',eta_search)
        print cmd
        os.system(cmd)

