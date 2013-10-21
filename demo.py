import os
import sys

opt_list = ['SGD','STG','RDA','Ada-FOBOS','Ada-RDA', 'AROW']

dataset = 'uci'

rootDir = '/home/matthew/work/Data/'

if dataset == 'MNIST':
    train_file = rootDir + 'MNIST/train-images-idx3-ubyte';
    train_label_file =rootDir + 'MNIST/train-labels-idx1-ubyte';
    test_file = rootDir + 'MNIST/t10k-images-idx3-ubyte';
    test_label_file = rootDir + 'MNIST/t10k-labels-idx1-ubyte';

    cmd_data = ' -i %s' %train_file + ' -t %s' %test_file + ' -il %s' %train_label_file + ' -tl %s' %test_label_file;
    cmd_data += ' -dt MNIST -d1 6 -d2 7'

elif dataset == 'uci':
    train_file = rootDir + 'uci/a6a'
    test_file = rootDir + '/uci/a6a.t'

    cmd_data = ' -i %s' %train_file + ' -t %s' %test_file

elif dataset == 'rcv1':
    train_file = rootDir + 'rcv1/rcv1.train' 
    test_file = rootDir + '/rcv1/rcv1.test'

    cmd_data = ' -i %s' %train_file + ' -t %s' %test_file

os.system('rm -fr ./result')

for opt in opt_list:
    cmd = 'python run_experiment.py %s' %opt 
    cmd += cmd_data
    print cmd
    os.system(cmd)

opt_list_file = './result/opt_list.txt' 
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
