#!/usr/bin/env python
"""dataset list"""

import sys
import os
import re
import random

import util

class DataSet(object):
    #constraint: only 
    __slots__ = ('root_dir','name','train_file','test_file', 'dim', 'class_num','lambda_list', 'l0_list')

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

        #set l0 list
        self.l0_list = [self.dim * 0.1 * x for x in range(1,10)]

    def __del__(self):
        self.del_rand_file()

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
    def set_lambda_list(self, l1_list):
        self.lambda_list = l1_list

    #set the feature selection rate
    def set_fs_rate(self, fs_rate):
        self.l0_list = [self.dim * x for x in fs_rate]


    #get the training cmd in the format of '-i -t '
    def get_train_cmd(self, rand_num, is_cache = True):
        if rand_num > 1:
            return util.get_train_cmd(self.train_file + '_rand',self.test_file, is_cache)
        else:
            return util.get_train_cmd(self.train_file,self.test_file, is_cache)

    def get_best_param(self, model):
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

    #merge different files together
    #return: merged file name
    def merge_files(self, fold_id_list):
        out_file = self.train_file + '_' + ''.join([str(item) for item in fold_id_list])

        wfh = open(out_file,'wb') 
        for fold_id in fold_id_list:
            in_file = '{0}_{1}'.format(self.train_file,fold_id)
            try:
                rfh = open(in_file, 'rb')
                lines = rfh.readlines()
                #remove empty lines
                while len(lines) > 0 and (lines[-1] == '\n' or lines[-1] == '\r\n'):
                    lines = lines[0:-1]
                #write the lines
                wfh.writelines(lines)
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
            lines = file.readlines()
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

aut = DataSet('aut')
dt_dict['aut'] = aut

#a9a = DataSet('a9a','a9a/a9a', 'a9a/a9a.t')
#dt_dict['a9a'] = a9a

