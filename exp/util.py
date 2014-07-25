#!/usr/bin/env python

import platform
import re

#get platform
def get_platform():
    return platform.system()

#definition of the result of training results
class ResultItem(object):
    __slots__ = ('train_error', 'test_error','train_time','test_time',
            'non_zero_num', 'sparse_rate','train_iter_error_list')

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
                    file_handler.write(str(val[k]) + ' ')
                file_handler.write('\n')
        except IOError as e:
            print "I/O error ({0}): {1}".format(e.errno,e.strerror)
            sys.exit()
        else:
            file_handler.close()

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
