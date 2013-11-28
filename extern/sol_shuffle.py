import random
import sys

if len(sys.argv) != 3:
    print 'usage: sol_shuffle.py input output_num'
    sys.exit()

filename = sys.argv[1]
num = int(sys.argv[2])
try:
    file = open(filename, 'rb')
    lines = file.readlines()
    if len(lines) == 0:
        print 'empty file'
        file.close()
        sys.exit()

    for k in range(0,num):
        random.shuffle(lines)
        wfile = open(filename + str(k), 'wb')
        wfile.writelines(lines)
        wfile.close()
except IOError as e:
    print "I/O error ({0}): {1}".format(e.errno, e.stderror)
    sys.exit()
else:
    file.close()
