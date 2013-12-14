import sys
import os
import time

def Usage():
    print 'Usage: libsvm2vw.py filename'

if len(sys.argv) != 2:
    Usage()
    sys.exit()

exename = '~/work/vw/vw_process'
filename = sys.argv[1]
output_filename = filename + '.vw'
tmp_filename = 'tmp.data'


start = time.time()
print 'change label +1 to 1'
cmd = 'cat %s' %filename + " | sed -e 's/^+1/1/' " + " > %s" %output_filename
os.system(cmd)
#transform to 
print 'transfrom libsvm into vw format'
start = time.time()
cmd = exename + ' %s' %output_filename + ' > %s' %tmp_filename
os.system(cmd)
print 'change label 0 to -1'
cmd = 'cat %s' %tmp_filename + " | sed -e 's/^0/-1/' " + " > %s" %output_filename
os.system(cmd)

end = time.time()
print('time elapsed: ' + str(end -start))

end = time.time()
print('time elapsed: ' + str(end -start))

os.system('rm -f %s' %tmp_filename)
