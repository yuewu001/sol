#!/usr/bin/env python

import platform
import re
import os

#get platform
def get_platform():
    return platform.system()

def get_train_cmd(train_file, test_file, is_cache):
    cmd = ' -i \"{0}\" -t \"{1}\" '.format(train_file, test_file)
    if is_cache == True:
        cmd += ' -c \"{0}\" -tc \"{1}\" '.format(train_file + '_cache', test_file + '_cache')

    cmd = cmd.replace('/',os.sep)
    return cmd

#definition of the result of training results
class ResultItem(object):
    __slots__ = ('train_error', 'test_error','train_time','test_time',
            'non_zero_num', 'sparse_rate')

    def __init__(self):
        for name in ResultItem.__slots__:
            self.__setattr__(name,[])

    def __setattr__(self, name, val):
        if name in ResultItem.__slots__:
            object.__setattr__(self,name,val)

    def GetValue(self, name):
        return self.__getattribute__(name)

    def __getattributre__(self, name):
        try:
            return object.__getattrbute__(self,name)
        except:
            return "value of %s" %name

    def Display(self):
        for name in ResultItem.__slots__:
            print name + ':\t' + str(self.__getattribute__(name))

    #add another result 
    def Add(self, other):
        for name in ResultItem.__slots__:
            val1 = self.__getattribute__(name)
            val2 = other.GetValue(name)

            if type(val2) is not list:
                self.__setattr__(name,  val1 + val2 )
            else:
                if type(val1) is not list or len(val1) == 0:
                    self.__setattr__(name,val2)
                else:
                    val = [val1[i] + val2[i] for i in range(0,len(val2))]
                    self.__setattr__(name,val)

    #divide by a value
    def Divide(self, divider):
        for name in ResultItem.__slots__:
            val = self.__getattribute__(name)

            if type(val) is not list:
                self.__setattr__(name,  val / divider )
            else:
                val = [val[i] / divider for i in range(0,len(val))]
                self.__setattr__(name,val)

    #append another result item
    def Append(self, other):
        for name in ResultItem.__slots__:
            val1 = self.__getattribute__(name)
            val2 = other.GetValue(name)

            if type(val2) is not list:
                raise ValueError('only list objects are suported!')
            else:
                if len(val2) == 0:
                    continue

                [val1.append(val2[i]) for i in range(0,len(val2))]

    #add a value to the give attribute
    def append_value(self, name, val):
        val_list = self.__getattribute__(name)
        val_list.append(val)
        self.__setattr__(name,val_list)

    #parse the ofs training result 
    #@param input_file: file contains the training result
    def parse_ofs_result(self, input_file):
        dec_pattern = "\d+\.?\d*"
        pattern_le = re.compile(r'Training error rate\s*:\s*(' + dec_pattern + ').*\s*')
        pattern_lt = re.compile(r'Training time\s*:\s*(' + dec_pattern + ').*\s*')
        pattern_nnz = re.compile(r'Number of NonZero weight\s*:\s*(' + dec_pattern + ').*\s*')
        pattern_sr = re.compile(r'Sparsification Rate\s*:\s*(' + dec_pattern + ').*\s*')
        pattern_te = re.compile(r'Test error rate\s*:\s*(' + dec_pattern + ').*\s*')
        pattern_tt = re.compile(r'Test time:\s*(' +dec_pattern + ')\s*s') 
        
        try:
            fh = open(input_file,'r')
            file_content = fh.read()

            tmp = pattern_le.findall (file_content)
            self.train_error    = [(float)(tmp[i]) for i in range(0,len(tmp))] 
            tmp = pattern_lt.findall (file_content)
            self.train_time     = [(float)(tmp[i]) for i in range(0,len(tmp))] 
            tmp = pattern_nnz.findall (file_content)
            self.non_zero_num   = [(float)(tmp[i]) for i in range(0,len(tmp))] 
            tmp = pattern_sr.findall (file_content)
            self.sparse_rate    = [(float)(tmp[i]) for i in range(0,len(tmp))] 
            tmp = pattern_te.findall (file_content)
            self.test_error     = [(float)(tmp[i]) for i in range(0,len(tmp))] 
            tmp = pattern_tt.findall (file_content)
            self.test_time      = [(float)(tmp[i]) for i in range(0,len(tmp))] 

        except IOError as e:
            print 'IO Error {0}: {1}'.format(e.errno, e.strerror)
        except Exception as e:
            print e
        else:
            fh.close()

    #get the result with different items in a group
    def get_result(self):
        item_num = len(self.GetValue('train_error'))
        result = [ResultItem() for k in range(0,item_num)]

        for name in ResultItem.__slots__:
            val = self.__getattribute__(name)
            if len(val) == 0:
                continue
            [result[k].append_value(name,val[k]) for k in range(0,item_num)]

        return result

    #save the result to local disk
    def save_result(self, output_file):
        open(output_file,'w').close()
        print 'save result to %s\n' %output_file
        try:
            file_handler = open(output_file,'w')

            #output header
            for name in ResultItem.__slots__:
                val = self.__getattribute__(name)
                if len(val) == 0:
                    continue
                file_handler.write(name + ' ')
            file_handler.write('\n')

            item_num = len(self.GetValue('train_error'))
            #output value
            for k in range(0,item_num):
                for name in ResultItem.__slots__:
                    val = self.__getattribute__(name)
                    if len(val) == 0:
                        continue
                    file_handler.write('%.2f' %val[k] + ' ')
                file_handler.write('\n')
        except IOError as e:
            print "I/O error ({0}): {1}".format(e.errno,e.strerror)
            sys.exit()
        else:
            file_handler.close()


#run the cv-train-algorithm
#@param train_file: training file
#@param test_file: testing file
#@param class_num: number of classes of the data
#@param param_config: parameter configuration
#@param model: model to train 
#@param config: cnfiguration to train the model
#@param output_file: output file to save the results
def run(train_file, test_file, class_num, param_config, model, config, output_file):
    #ofs executable
    if get_platform() == 'Windows':
        ofs_exe = r'..\install\bin\SOL.exe'
    else:
        ofs_exe = '../install/bin/SOL'

    #evaluate the result
    cmd_postfix = ' >> %s' %output_file

    dt_cmd = get_train_cmd(train_file, test_file, config['cache'])
    if class_num > 2:
        if model == 'DAROW':
            loss_cmd = ' -cn %d -loss MaxScoreSquaredHinge ' %(class_num)
        else:
            loss_cmd = ' -cn {0} -loss {1} '.format(class_num, config['mc_loss'])
    else:
        if model == 'DAROW':
            loss_cmd = ' -cn 2 -loss SquaredHinge ' 
        else:
            loss_cmd = ' -cn 2 -loss {0} '.format(config['bc_loss'])

    norm_cmd = ' -norm ' if config['norm'] == True else '' 

    cmd_prefix  = ofs_exe + dt_cmd + loss_cmd + norm_cmd  + ' -m %s ' %model
    if 'passes' in config:
        cmd_prefix += ' -passes %d ' %config['passes']

    cmd = cmd_prefix + param_config + cmd_postfix
    print cmd
    os.system(cmd)

    #parse the result
    result = ResultItem()
    result.parse_ofs_result(output_file)

    return result

#test code
#a = ResultItem()
#b = ResultItem()
#b.parse_ofs_result('tmp.txt')
#a.Add(b)
#a.Add(b)
#a.Divide(2)
#a.Display()
#
#a.save_result('output.txt')

