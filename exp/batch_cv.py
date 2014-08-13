#!/usr/bin/env python
"""bach cross validation"""
import CV
import dataset

model_list = ['SOFS','FOFS','PET','PreSelOGD']

ds_list = ['rcv1','news','url']
ds_list = ['synthetic_10K', 'synthetic_20K', 'relathe','pcmac','basehock','ccat','aut','real-sim'] 

fold_num = 5

const_eta_search = '0.03125:2.0:32'
eta_search = '0.25:2.0:256'
delta_search = '0.03125:2.0:32'
r_search = '0.25:2.0:256'
delta_ofs_search = '0.0003125:2:0.32'

for dt in ds_list:
    for model in model_list:
        print '----------------------------------------------'
        print 'Cross validation on %s' %dt + ' with %s' %model
        print '----------------------------------------------'

        real_model = ''
        if model == 'SOFS':
            real_model = model
            model = 'DAROW'
        elif model == 'PET' or 'PreSelOGD' in model:
            real_model = model
            model = 'SGD'

        if  model == 'CW_TG' or model == 'CW_RDA' or model == 'DAROW':
            param_config = ' {0} {1}'.format('-r',r_search)
        elif 'Ada' in model: 
            param_config = ' {0} {1}'.format('-delta',delta_search)
            param_config += ' {0} {1}'.format('-eta',const_eta_search)
        elif model == 'FOFS':
            param_config = ' {0} {1}'.format('-delta',delta_ofs_search)
            param_config += ' {0} {1}'.format('-eta',const_eta_search)
        else:
            param_config = ' {0} {1}'.format('-eta',eta_search)

        cv = CV.CV(dt,model,fold_num,param_config)
        cv.run()

        if real_model != '':
            src_out_file = '{0}/cv/cv_{1}_result.txt'.format(dt,model)
            dst_out_file = '{0}/cv/cv_{1}_result.txt'.format(dt,real_model)
            with open(src_out_file,'r') as rfh:
                content = rfh.read()
                with open(dst_out_file,'w') as wfh:
                    wfh.writelines(content)

    dataset.dt_dict[dt].del_split_files(fold_num)
