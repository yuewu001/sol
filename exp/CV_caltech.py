#!/usr/bin/env python
"""Cross validation"""

import sys
import os
import dataset
import util
import search_space 

#extra command sent to SOL
model_config = {
'cache':True,
'norm':False,
'bc_loss':'Hinge',
'mc_loss':'MaxScoreHinge',
'passes':10
}

#definition of cross validation class
class CV(object):
    #member definition
    __slots__ = ('dataset','model','fold_num','dst_folder','search_space','result','min_result_key', 'min_result_val')

    def __init__(self, dt_name, model, fold_num, param_space):
        if dt_name in dataset.dt_dict:
            self.dataset = dataset.dt_dict[dt_name]
        else:
            raise ValueError('dataset {0} does not exist!'.format(dt_name))

        #for cv, no feature selection or l1 regularization is required
        self.dataset.set_lambda_list([0])
        self.dataset.set_fs_rate([1])

        self.model = model

        self.fold_num = int(fold_num)
        if self.fold_num < 2 :
            raise ValueError('fold number must bigger than 2!')

        self.dst_folder = self.dataset.name + '/cv'

        #generate the search space
        self.search_space = search_space.SearchSpace(param_space)

        #search_space.SeachSpace(param_space)

        self.result = util.ResultItem()

    #cross validation on one fold of data
    #@param fold_id: fold id that is used as test data
    def __train(self, test_fold_id):
        result_file = '{0}/{1}_result_{2}.txt'.format(self.dst_folder,self.model, test_fold_id)
        if os.path.exists(result_file):
            os.remove(result_file)

        #merge files
        merge_list = [i for i in range(0,self.fold_num) if i != test_fold_id]

        train_file = self.dataset.merge_files(merge_list)
        test_file = self.dataset.train_file + '_' + str(test_fold_id)

        #train
        for k in range(0, self.search_space.size):
            param_cmd  = self.search_space.get_param_cmd(k)
            #does not select features
            param_cmd += ' -k %d ' %(self.dataset.dim)

            result_once = util.run(train_file, test_file, self.dataset.class_num,
                    param_cmd, self.model, model_config, result_file)

        self.result.Add(result_once)

        #delete temp file
        self.dataset.del_file_and_cache(train_file)
        self.dataset.del_cache(test_file)

#    #cross validation
    def run(self):
        #save result
        final_file = self.dst_folder +'/cv_{0}_result.txt'.format(self.model)
        if os.path.exists(final_file):
            print 'file already exists, skip cross validation'
            return

        #split the dataset
        self.dataset.split(self.fold_num)

        if os.path.exists(self.dst_folder) == False:
            os.makedirs(self.dst_folder)

        #cross validation
        for test_fold_id in range(0,self.fold_num):
            print '---------------------------'
            print 'Cross Validation: Folder %d' %test_fold_id
            print '---------------------------'
            self.__train(test_fold_id)

        #average the result
        self.result.Divide(self.fold_num)

        #merge param cmd and result
        self.result = self.result.get_result()
        param_result = {}
        for k in range(0, self.search_space.size):
            param_cmd  = self.search_space.get_param_cmd(k)
            param_result[param_cmd] = self.result[k]

        #find the min test error rate, 
        self.min_result_key = ''
        self.min_result_val = util.ResultItem()

        for k,v in param_result.iteritems():
            if len(self.min_result_val.test_error) == 0 or self.min_result_val.test_error[0] >  v.test_error[0]:
                self.min_result_val = v
                self.min_result_key = k

        print '\ncross validation result: '
        print '\tparameters:\t{0}\n\ttrain error:\t{1}\n\ttest error:\t{2}'.format(
                self.min_result_key, self.min_result_val.train_error[0], self.min_result_val.test_error[0])

        with open(final_file,'w') as wfh:
            wfh.write('Best Result: {0}:\t{1}\t{2}\n'.format(self.min_result_key,
                self.min_result_val.train_error[0],
                self.min_result_val.test_error[0]))

            for k,v in param_result.iteritems():
                wfh.write('%s:\t%.2f\t%.2f\n' %(k,v.train_error[0], v.test_error[0]))

        print '\ncross validation result written to %s' %final_file

if __name__ == '__main__':
    handler = CV('aut','Ada_RDA','3','-eta 0.5:10:12 -delta 0.5:2:1')
    handler.run()
