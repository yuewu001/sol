#!/usr/bin/env python
"""shuffle a dataset"""
import random
import sys

def sol_shuffle(filename, out_filename):
    try:
        file = open(filename, 'rb')
        lines = file.readlines()
        if len(lines) == 0:
            print 'empty file'
            file.close()
            sys.exit()
        if lines[-1][-1] != '\n':
            lines[-1]+='\n'

        random.shuffle(lines)
        wfile = open(out_filename, 'wb')
        wfile.writelines(lines)
        wfile.close()
    except IOError as e:
        print "I/O error ({0}): {1}".format(e.errno, e.strerror)
        sys.exit()
    else:
        file.close()
