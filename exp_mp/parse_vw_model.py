import sys
import re

if len(sys.argv) != 3:
    print 'Usage parse_vw_model model_file dim'
    sys.exit()

model_file = sys.argv[1]
dim = int(sys.argv[2])

int_pattern = "(\d+)"
dec_pattern = "([+-]?\d+\.?\d*)"
pattern = re.compile(r'' + int_pattern + ':' + dec_pattern) 

model_size = 0
try:
    file_handle = open(model_file,'r')
    while True:
        line = file_handle.readline()
        if not line:
            break;

        if pattern.match(line) != None:
            model_size += 1

    #result_list = pattern.findall(open(model_file,'r').read())

except IOError as e:
    print "I/O error ({0}): {1}".format(e.errno,e.strerror)
    sys.exit()
else:
    file_handle.close()

print 'Sparsification Rate: %.2f%%' %(float(model_size) * 100.0 / dim) 
