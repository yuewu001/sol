#This script is to run experiment automatically to test the performance of the algorithm

import os
import time
import sys
import re

import util

def run(dataset,model_config, param_config, output_file):
    if util.get_platform() == 'Windows':
        mrmr_exe = '../extern/mRMR/mrmr_win32.exe'
        converter_exe = '../install/bin/Converter.exe'
        sol_exe = r'..\install\bin\SOL.exe'
    else:
        mrmr_exe = '../extern/mRMR/mrmr_redhat_32'
        converter_exe = '../install/bin/Converter'
        sol_exe = '../install/bin/SOL'

    dst_folder = dataset.name + '/mRMR'
    if os.path.exists(dst_folder) == False:
        os.makedirs(dst_folder)

    result = util.ResultItem()

    data_dim = dataset.dim
    data_num = dataset.data_num

    #bs_list = l1_def.get_lambda_list(ds,'mRMR')

    sel_feat_num_list = [x for x in dataset.l0_list if x <= 500]
    
    #clear the file if it already exists
    open(output_file,'w').close()

    result_all = util.ResultItem()
    for sel_feat_num in sel_feat_num_list:
        raw_model_file = dst_folder + '/raw_model_%d' %sel_feat_num
        model_file = dst_folder + '/model_%d' %sel_feat_num

        #run mRMR

        mrmr_train_time  = 0
        #prepare training data
        if os.path.exists(raw_model_file) == False:
            train_file = dataset.get_train_file(model_config['rand_num'])
            csv_train_file =  train_file + '.csv'
            if os.path.exists(csv_train_file) == False:
                #convert data
                print 'convert data'
                cmd = converter_exe + ' -i %s' %train_file + ' -o %s' %csv_train_file
                cmd += ' -st libsvm -dt csv'
                cmd = cmd.replace('/',os.sep)
                print cmd
                os.system(cmd)


            prev_cmd = mrmr_exe + ' -v %d' %data_dim + ' -t 0.5 -i %s' %csv_train_file 
            cmd = prev_cmd + ' -n %d' %sel_feat_num + ' > %s' %raw_model_file
            cmd = cmd.replace('/',os.sep)
            print cmd
            start_time =time.time()
            os.system(cmd)
            end_time = time.time()

            #parse learning time
            mrmr_train_time = (float)(end_time - start_time)

        if os.path.exists(model_file) == False:
            #parse result
            parse_model_file(raw_model_file,model_file);

        #train with OGD
        dt_cmd = dataset.get_train_cmd(model_config['rand_num'],model_config['cache'])
        if dataset.class_num > 2:
            loss_cmd = ' -cn {0} -loss {1} '.format(dataset.class_num, model_config['mc_loss'])
        else:
            loss_cmd = ' -cn 2 -loss {0} '.format(model_config['bc_loss'])

        norm_cmd = ' -norm ' if model_config['norm'] == True else '' 

        cmd_prefix  = sol_exe + dt_cmd + loss_cmd + norm_cmd  + ' -m PreSelOGD '  + param_config

        if 'passes' in model_config:
            cmd_prefix += ' -passes %d ' %model_config['passes']

        cmd_postfix = ' >> %s' %output_file

        cmd = cmd_prefix + ' -im %s' %model_file + cmd_postfix

        print cmd
        os.system(cmd)

        #parse the result
        result = util.ResultItem()
        result.parse_ofs_result(output_file)
        result.train_time[0] += mrmr_train_time

        print '\nTraining Result: '
        result.Display()
        print '\n'

        result_all.Append(result)

    return result_all


def parse_model_file(model_file,parse_file):
    print 'parse model file of mRMR%s\n' %model_file
    c_feat = []
    pattern = re.compile(r'(\S*)\s*')
    is_begin = False
    try:
        file_handler = open(model_file,'r')
        while True:
            line = file_handler.readline()
            line = line.strip()
            if is_begin == True and len(line) == 0:
                break
            if line == '*** mRMR features ***':
                line = file_handler.readline()
                is_begin = True
                continue
            if (is_begin == False):
                continue
            result_list = pattern.findall(line)
            c_feat.append(int(result_list[1]))
    except IOError as e:
        print "I/O error ({0}): {1}".format(e.errno,e.strerror)
        sys.exit()
    else:
        file_handler.close()
        print 'feature number %d' %(len(c_feat))
    #write c_feat into file
    try:
        file_handler = open(parse_file,'wb')
    
        for k in range(0,len(c_feat)):
            file_handler.write('%d\n' %c_feat[k])
    except IOError as e:
        print "I/O error ({0}): {1}".format(e.errno,e.strerror)
        sys.exit()
    else:
        file_handler.close()
    return c_feat
    


