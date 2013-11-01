import os
import sys

opt_list = ['SGD','STG','RDA','Ada-FOBOS','Ada-RDA', 'AROW', 'SSAROW', 'ASAROW', 'vw']
#opt_list = ['vw']

rootDir = '/home/matthew/work/Data/'
if len(sys.argv) == 2:
    dataset = sys.argv[1]
else:
    dataset = 'rcv1'

if dataset == 'a6a':
    train_file = rootDir + 'uci/a6a'
    test_file = rootDir + '/uci/a6a.t'

    cmd_data = ' -i %s' %train_file + ' -t %s' %test_file + ' -passes 5'

elif dataset == 'a9a':
    train_file = rootDir + 'uci/a9a'
    test_file = rootDir + 'uci/a9a.t'

    cmd_data = ' -i %s' %train_file + ' -t %s' %test_file  + ' -passes 20'

elif dataset == 'rcv1':
    train_file = rootDir + 'rcv1/rcv1.train' 
    test_file = rootDir + '/rcv1/rcv1.test'

    cmd_data = ' -i %s' %train_file + ' -t %s' %test_file + ' '

elif dataset == 'kdda':
    train_file = rootDir + 'kdda/algebra/kdda'
    test_file = rootDir + 'kdda/algebra/kdda.t'

    cmd_data = ' -i %s' %train_file + ' -t %s' %test_file + ' '
elif dataset == 'epsilon':
    train_file = rootDir + 'epsilon/epsion_normalized'
    test_file = rootDir + 'epsilon/epsion_normalized.t'

    cmd_data = ' -i %s' %train_file + ' -t %s' %test_file + ' '

else:
    print 'unrecoginized dataset'
    sys.exit()


cache_train_file = train_file + '_cache'
cache_test_file = test_file + '_cache'
cmd_data += ' -c %s' %cache_train_file
cmd_data += ' -tc %s' %cache_test_file

dst_folder = './result/%s' %dataset 

#analyze dataset
dataset_info_file = train_file + '_info.txt'
if os.path.exists(dataset_info_file) == False:
    print 'analyze dataset'
    cmd = '../analysis %s' %train_file +' >> %s' %dataset_info_file
    os.system(cmd)

#train model
for opt in opt_list:
    print '-----------------------------------'
    print 'Experiment on %s' %opt
    print '-----------------------------------'
    if opt == 'vw':
        cmd = 'python run_vw.py %s' %dst_folder + ' %s' %train_file 
        cmd += ' %s'  %cache_train_file + ' %s' %test_file + ' %s' %cache_test_file
        os.system(cmd)

    else:
        cmd = 'python run_experiment.py %s' %opt  + ' %s' %dst_folder
        if opt == 'RDA': 
            cmd += ' -eta 100'
        else:
            cmd += ' -eta 1'

        cmd += cmd_data
        os.system(cmd)

opt_list_file = './%s' %dst_folder + '/opt_list.txt' 
#clear the file if it already exists
open(opt_list_file,'w').close()


try:
    file_handle = open(opt_list_file,'w')
    for opt in opt_list:
        file_handle.write(opt + '.txt\n')
except IOError as e:
    print "I/O error ({0}): {1}".format(e.errno,e.strerror)
    sys.exit()
else:
    file_handle.close()
