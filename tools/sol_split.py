#!/usr/bin/env python
import sys

if len(sys.argv) != 5:
    print 'usage: sol_split.py input output_1 output_2 num'
    sys.exit()

filename = sys.argv[1]
outname1 = sys.argv[2]
outname2 = sys.argv[3]
num = int(sys.argv[4])
print num
try:
    file = open(filename, 'rb')
    outfile1 = open(outname1, 'wb')
    outfile2 = open(outname2, 'wb')
    index = 0
    while True:
		line = file.readline()
		if len(line) == 0:
			break
		if index < num:
			outfile1.write(line)
		else:	
			outfile2.write(line)
		index += 1
except IOError as e:
    print "I/O error ({0}): {1}".format(e.errno, e.stderror)
    sys.exit()
else:
    file.close()
    outfile1.close()
    outfile2.close()
