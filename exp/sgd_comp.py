import os
import sys

opt_list = ['SGD','vw']#, 'liblinear']
#opt_list = ['liblinear']

is_cache = True

rootDir = '/home/matthew/data/'
if len(sys.argv) == 2:
    dataset = sys.argv[1]
else:
    dataset = 'rcv1'


if dataset == 'a6a':
    train_file = rootDir + 'uci/a6a'
    test_file = rootDir + '/uci/a6a.t'

    cmd_data = ' -i %s' %train_file + ' -t %s' %test_file + ' -eta 0.01'
elif dataset == 'a9a':
    train_file = rootDir + 'uci/a9a'
    test_file = rootDir + 'uci/a9a.t'

    cmd_data = ' -i %s' %train_file + ' -t %s' %test_file + ' -eta 0.01'
elif dataset == 'rcv1':
    train_file = rootDir + 'rcv1/rcv1.train' 
    test_file = rootDir + '/rcv1/rcv1.test'

    cmd_data = ' -i %s' %train_file + ' -t %s' %test_file + ' -eta 1.0'
else:
    print 'unrecoginized dataset'
    sys.exit()

dst_folder = '%s' %dataset + '_sgd_comp'

cache_train_file = train_file + '_cache'
cache_test_file = test_file + '_cache'
cmd_data += ' -c %s' %cache_train_file
cmd_data += ' -tc %s' %cache_test_file

#analyze dataset
dataset_info_file = train_file + '_info.txt'
if os.path.exists(dataset_info_file) == False:
    print 'analyze dataset'
    cmd = './analysis %s' %train_file +' >> %s' %dataset_info_file
    os.system(cmd)

#train model
if is_cache == False:
    for opt in opt_list:
        if opt == 'vw':
            cmd = 'python run_vw.py %s' %dst_folder + ' %s' %train_file 
            cmd += ' nocache'  + ' %s' %test_file + ' nocache' 
            cmd += ' no_l1 no_cache'
            print cmd
            os.system(cmd)
        elif opt == 'liblinear':
            cmd = 'python run_liblinear.py %s' %dst_folder + ' ll_model'
            cmd += ' %s' %train_file + ' %s' %test_file
            print cmd
            os.system(cmd)
        elif opt == 'SGD':
            cmd = 'python run_experiment.py %s' %opt  + ' %s' %dst_folder
            cmd += cmd_data
            cmd += ' no_l1'
            print cmd
            os.system(cmd)
else:
    for opt in opt_list:
        if opt == 'vw':
            cmd = 'python run_vw.py %s' %dst_folder + ' %s' %train_file 
            cmd += ' %s' %cache_train_file + ' %s' %test_file + ' %s' %cache_test_file 
            cmd += ' no_l1'
            print cmd
            os.system(cmd)
        elif opt == 'SGD':
            cmd = 'python run_experiment.py %s' %opt  + ' %s' %dst_folder
            cmd += cmd_data
            cmd += ' no_l1'
            print cmd
            os.system(cmd)
