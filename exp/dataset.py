#!/usr/bin/env python
"""dataset list"""

import sys
import os
import re
import random

import util

class DataSet(object):
    #constraint: only 
    __slots__ = ('root_dir','name','train_file','test_file', 'dim', 'data_num',
            'class_num','lambda_list', 'lambda_dict','l0_list','c_list','mrmr_l0_list')

    root_dir = 'D:/Data/libsvm/'

    def __init__(self, dt_name, train_file = '', test_file = ''):
        self.name = dt_name
        if train_file == '':
            self.train_file = self.root_dir + '{0}/{0}_train'.format(dt_name, dt_name)
        else:
            self.train_file = self.root_dir + train_file
        if test_file == '':
            self.test_file = self.root_dir + '{0}/{0}_test'.format(dt_name,dt_name)
        else:
            self.test_file = self.root_dir + test_file

        if os.path.exists(self.train_file) == False or os.path.exists(self.test_file) == False:
            raise Exception('train file or test file not found!')

        #analyze the dataset
        self.__analyze_dataset()

        #set the lambda list
        self.lambda_list = [pow(10,x) for x in range(-8,0,1)]

        self.lambda_dict = {}

        #set l0 list
        self.set_fs_rate([0.005,0.01,0.025,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
        #self.l0_list = [self.dim * 0.1 * x for x in range(1,10)]

        #set mrmr list
        #self.set_mrmr_l0_list([self.dim * x for x in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.975, 0.99,0.995]])
        self.set_mrmr_l0_list([50,100,150,200,250,300,350,400,450,500])

        #set c_list: for liblinear
        self.c_list = [pow(10,x) for x in range(-3,3,1)]
        self.c_list = [0.015625,0.03125,0.0625,0.125,0.25,0.5,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,9182,18364]

    def __del__(self):
        #self.del_rand_file()
        pass

    #delete rand file and  the corresponding cache file
    def del_rand_file(self):
        rand_file = self.train_file + '_rand'

        self.del_file_and_cache(rand_file)


    #delete rand file and  the corresponding cache file
    def del_file_and_cache(self, filename):
        self.__del_file(filename)
        self.del_cache(filename)

    def del_cache(self, filename):
        cache_file = filename + '_cache'
        self.__del_file(cache_file)

    def __del_file(self, filename):
        if os.path.exists(filename):
            os.remove(filename)

    #set the lambda list
    def set_lambda_list0(self, l1_list):
        self.lambda_list = l1_list

    def set_lambda_list(self, algo, l1_list):
        self.lambda_dict[algo] = l1_list

    def get_lambda_list(self, algo):
        if algo in self.lambda_dict:
            return self.lambda_dict[algo]
        else:
            return self.lambda_list

    #set the feature selection rate
    def set_fs_rate(self, fs_rate):
        self.l0_list = [int(self.dim * x) for x in fs_rate]

    #set the feature selection rate
    def set_fs_num(self, fs_num):
        self.l0_list = [x for x in fs_num if (x > 0 and x < self.dim)]

    #set number of selected features for mrmr
    def set_mrmr_l0_list(self, mrmr_num):
        self.mrmr_l0_list = [x for x in mrmr_num if (x > 0 and x < self.dim)]

    #set the c parameter for liblinear
    def set_c_list(self,c_list):
        self.c_list = c_list

    #get the train file
    def get_train_file(self, rand_num):
        if rand_num > 1:
            return self.train_file + '_rand'
        else:
            return self.train_file

    #get the training cmd in the format of '-i -t '
    def get_train_cmd(self, rand_num, is_cache = True):
        if rand_num > 1:
            return util.get_train_cmd(self.train_file + '_rand',self.test_file, is_cache)
        else:
            return util.get_train_cmd(self.train_file,self.test_file, is_cache)

    def get_best_param(self, model):
        if '+' in model:
            cv_file = '{0}/cv/cv_{1}_result.txt'.format(self.name, model.split('+')[0])
        else:
            cv_file = '{0}/cv/cv_{1}_result.txt'.format(self.name, model)
        dec_pattern = "\d+\.?\d*"
        pattern = re.compile(r'(?<=Best Result:).+(?=:)'.format(dec_pattern))

        try:
            cv_file = cv_file.replace('/',os.sep)
            fh = open(cv_file,'r')
            file_content = fh.readline()

            result = pattern.findall (file_content)
            if len(result) == 0:
                raise 'cross validation file is incorrect!'

        except IOError as e:
            print 'Error {0}: {1}'.format(e.errno, e.strerror)
        else:
            fh.close()

        return result[0]

    #analyze the dataset to obtain dim and class number
    def __analyze_dataset(self):
        info_file = self.train_file + '_info.txt'
    
        #if not analyzed before, analyze
        if os.path.exists(info_file) == False :
            if util.get_platform() == 'Windows':
                exe_name = r'..\install\bin\data_analysis.exe'
            else:
                exe_name = r'../install/bin/data_analysis'

            if os.path.exists(exe_name) == False:
                print 'analyze executable not exist!'
                sys.exit()
            print 'calculate dimension of %s' %self.train_file
            cmd = '{0} -i \"{1}\" -st libsvm >> {2}'.format(exe_name,self.train_file,info_file)
            print cmd
            os.system(cmd)

        #parse data num
        pattern = re.compile(r'data number\s*:\s*(\d+)')
        result_list = pattern.findall(open(info_file,'r').read())
        if len(result_list) != 1:
            print result_list
            print 'parse failed'
            sys.exit()
    
        self.data_num = (int)(result_list[0])

        #parse dimension
        pattern = re.compile(r'dimension\s*:\s*(\d+)')
        result_list = pattern.findall(open(info_file,'r').read())
        if len(result_list) != 1:
            print result_list
            print 'parse failed'
            sys.exit()
    
        self.dim = (int)(result_list[0])

        #parse class number
        pattern = re.compile(r'class num\s*:\s*(\d+)')
        result_list = pattern.findall(open(info_file,'r').read())
        if len(result_list) != 1:
            print result_list
            print 'parse failed'
            sys.exit()
    
        self.class_num = (int)(result_list[0])


    def split(self, fold_num):
        print 'split files...'
        in_filename = self.train_file
        #count line number
        line_num = 0
        try:
            file = open(in_filename, 'rb')
            while True:
                line = file.readline()
                if len(line) == 0:
                    break
                line_num += 1
        except IOError as e:
            print "I/O error ({0}): {1}".format(e.errno, e.strerror)
            sys.exit()
        else:
            file.close()
        print line_num, " lines in file"

        #get the file length of splitted files
        split_file_len_list = [line_num / fold_num for i in range(0,fold_num - 1)]
        #last file length needs special care
        split_file_len_list.append(line_num - (line_num / fold_num) * (fold_num - 1))

        #split the files
        with open(in_filename,'rb') as rfh:
            for i in range(0,fold_num):
                out_filename = '{0}_{1}'.format(in_filename,i) 
                with open(out_filename,'wb') as wfh:
                    for j in range(0,split_file_len_list[i]):
                        line = rfh.readline()
                        wfh.write(line)

    #delete the splitted files
    def del_split_files(self, fold_num):
        in_filename = self.train_file
        for i in range(0,fold_num):
            out_filename = '{0}_{1}'.format(in_filename,i) 
            self.__del_file(out_filename)

    #merge different files together
    #return: merged file name
    def merge_files(self, fold_id_list):
        print 'merge files...'
        out_file = self.train_file + '_' + ''.join([str(item) for item in fold_id_list])

        wfh = open(out_file,'wb') 
        for fold_id in fold_id_list:
            print fold_id
            in_file = '{0}_{1}'.format(self.train_file,fold_id)
            try:
                rfh = open(in_file, 'rb')
                while True:
                    line = rfh.readline()
                    #remove empty lines
                    if len(line) == 0 or line == '\n' or line == '\r\n':
                        break
                    #write the line
                    wfh.write(line)
            except IOError as e:
                print 'Error {0}: {1}'.format(e.errno, e.strerror)
            else:
                rfh.close()
        return out_file

    #shuffle a data file
    def shuffle_file(self):
        #delete previous cache (rand file and its cache file)
        self.del_rand_file()

        in_filename = self.train_file
        out_filename = in_filename + '_rand'
        try:
            file = open(in_filename, 'rb')
            lines = []
            while True:
                line = file.readline()
                if len(line) == 0:
                    break
                lines.append(line)

            if len(lines) == 0:
                print 'empty file'
                file.close()
                sys.exit()

            #make sure that the last line ends with '\n'
            if lines[-1][-1] != '\n':
                lines[-1]+='\n'

            #remove empty lines
            while len(lines) > 0 and  (lines[-1] == '\n' or lines[-1] == '\r\n'):
                lines = lines[0:-1]

            random.shuffle(lines)
            wfile = open(out_filename, 'wb')
            wfile.writelines(lines)
            wfile.close()
        except IOError as e:
            print "I/O error ({0}): {1}".format(e.errno, e.strerror)
            sys.exit()
        else:
            file.close()

#initialize dataset
dt_dict = {}

##synthetic
#synthetic_10K = DataSet('synthetic_10K','synthetic_10K/synthetic_train', 'synthetic_10K/synthetic_test')
#synthetic_10K.set_c_list([0.00001,0.0001,0.00025,0.0005,0.00075,0.001,0.01,0.015,0.016,0.017,0.018,0.019,0.02])
#synthetic_10K.set_fs_num([50,60,70,80,90,100,120,140,160,180,200])
#synthetic_10K.set_mrmr_l0_list([50,60,70,80,90,100,120,140,160,180,200])
#dt_dict['synthetic_10K'] = synthetic_10K
#
#synthetic_20K = DataSet('synthetic_20K','synthetic_20K/synthetic_train','synthetic_20K/synthetic_test')
#synthetic_20K.set_c_list([0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0008,0.0007,0.0009,0.001,0.01,0.012,0.013,0.014,0.015,0.016,0.018,0.02])
#synthetic_20K.set_fs_num([150,160,170,180,190,200,220,240,260,280,300,400])
#synthetic_20K.set_mrmr_l0_list([150,160,170,180,190,200,220,240,260,280,300,400])
#dt_dict['synthetic_20K'] = synthetic_20K
#
#synthetic_1B = DataSet('synthetic_1B','synthetic_1B/synthetic_train','synthetic_1B/synthetic_test')
#synthetic_1B.set_fs_num([500])
#dt_dict['synthetic_1B'] = synthetic_1B
#
##media scale data
#relathe = DataSet('relathe')
#dt_dict['relathe'] = relathe
#
#pcmac = DataSet('pcmac')
#dt_dict['pcmac'] = pcmac
#
#basehock = DataSet('basehock')
#dt_dict['basehock'] = basehock
#
#ccat = DataSet('ccat')
#dt_dict['ccat'] = ccat
#
#aut = DataSet('aut')
#dt_dict['aut'] = aut
#
#real_sim = DataSet('real-sim')
#dt_dict['real-sim'] = real_sim
#
##large scale data
#news = DataSet('news')
#news.set_fs_rate([0.005,0.01,0.025,0.05,0.1,0.2])
#dt_dict['news'] = news
#
#rcv1  = DataSet('rcv1')
#rcv1.set_fs_rate([0.005,0.01,0.025,0.05,0.1,0.2])
#dt_dict['rcv1'] = rcv1
#
url = DataSet('url')
url.set_fs_rate([0.005,0.01,0.025,0.05,0.1,0.2])
url.set_lambda_list('CW_TG', [0,1e-8,1e-7,5e-7, 
    1e-6, 2e-6,3e-6,4e-6,5e-6,6e-6,7e-6,8e-6,9e-6,
    1e-5,2e-5,4e-5,6e-5,8e-5,
    1e-4,2e-4,4e-4,6e-4,8e-4,
    1e-3,2.5e-3,5e-3,7.5e-3,
    1e-2,2.5e-2,5e-2,7.5e-2,1e-1])

url.set_lambda_list('CW_RDA', [0,1e-7,
    1e-6,2.5e-6,5e-6,7.5e-6,1e-5,2.5e-5,5e-5,7.5e-5,
    1e-4,2.5e-4,5e-4,7.5e-4,1e-3,2.5e-3,5e-3,7.5e-3,
    1e-2,2.5e-2,5e-2,7.5e-2,1e-1])

url.set_lambda_list('STG', [0,1.25e-6,1.5e-6,1.75e-6,
    2.5e-6,5e-6, 1e-5,1.5e-5,2.5e-5,5e-5,7.5e-5,
    1e-4,2.5e-4,5e-4,7.5e-4,1e-3,2.5e-3,5e-3, 1e-2,2.5e-2])

url.set_lambda_list('FOBOS', url.get_lambda_list('STG'))

dt_dict['url'] = url
#
#caltech = DataSet('caltech_new','caltech/caltech.train.libsvm','caltech/caltech.test.libsvm')
#caltech.set_c_list([0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.06])
##caltech.set_fs_num([50,60,70,80,90,100,150,200,250,300,350,400,450,500,550,600,700,800,900,1000,1500,2000,2500,3000,3500])
#caltech.set_fs_num([1000,2000,3000,4000,5000,6000,7000,8000,9000])
#caltech.set_mrmr_l0_list([50,60,70,80,90,100,150,200,250,300,350,400,450,500])
#dt_dict['caltech_new'] = caltech
