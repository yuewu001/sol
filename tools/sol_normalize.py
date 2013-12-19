#!/usr/bin/env python
r"""normalze the data to desired format
+1 -> 1, 0 > -11
"""

import sys

def Usage():
    print 'Usage: sol_normalize in_file out_file'

if len(sys.argv) != 3:
    Usage()
    sys.exit()

in_filename = sys.argv[1]
out_filename = sys.argv[2]

line_num = 0
try:
    in_file = open(in_filename, "rb")
    out_file = open(out_filename, "wb")

    while True:
        line = in_file.readline()
        if len(line) == 0:
            break;
        line_num += 1
        if line.startswith("0"):
            line = "-1" + line[1:]
        elif line.startswith("+1"):
            line = "1" + line[2:]
        if line_num % 10000 == 0:
            print str(line_num) + 'lines processed'

        out_file.write(line)
except IOError as e:
    print "IO Error ({0}:{1})".format(e.errno, e.stderror)
else:
    in_file.close()
    out_file.close()
