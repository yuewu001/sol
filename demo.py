import os
import sys

#opt_list = ['SGD','STG','RDA','Ada-FOBOS','Ada-RDA', 'AROW']
opt_list = ['SGD','STG','RDA','Ada-FOBOS','Ada-RDA', 'AROW']
#train_file = '/home/matthew/work/data/rcv1.train'
#test_file = '/home/matthew/work/data/rcv1.test'

train_file = '/home/matthew/work/data/MNIST/train-images-idx3-ubyte';
train_label_file ='/home/matthew/work/data/MNIST/train-labels-idx1-ubyte';
test_file = '/home/matthew/work/data/MNIST/t10k-images-idx3-ubyte';
test_label_file = '/home/matthew/work/data/MNIST/t10k-labels-idx1-ubyte';


os.system('rm -fr ./result')

for opt in opt_list:
    cmd = 'python run_experiment.py %s' %opt 
    cmd += ' -i %s' %train_file + ' -t %s' %test_file + ' -il %s' %train_label_file + ' -tl %s' %test_label_file;
    cmd += ' -dt MNIST -d1 6 -d2 7'
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
