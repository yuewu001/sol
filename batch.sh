#########################################################################
# File Name: batch.sh
# Author: Yue Wu
# mail: yuewu@outlook.com
# Created Time: Tue 15 Oct 2013 09:53:16 AM PDT
#########################################################################
#!/bin/bash

./test -eta 1 >> rda.txt
./test -l1 1e-8 -eta 1 >> rda.txt
./test -l1 1e-7 -eta 1 >> rda.txt
./test -l1 1e-6 -eta 1 >> rda.txt
./test -l1 1e-5 -eta 1 >> rda.txt
./test -l1 1e-4 -eta 1 >> rda.txt
./test -l1 1e-3 -eta 1 >> rda.txt
