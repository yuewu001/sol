#!/usr/bin/evn python
'''This file defines the path to executables, users may need to change the
paths'''

import platform

#windows
if platform.system() == "Windows":
    SOL_exe_name = r'..\install\bin\SOL.exe'
    vw_exe_name = 'vw'
    liblinar_train_exe_name = r'..\extern\liblinear\train.exe'
    liblinar_test_exe_name = r'..\extern\liblinear\train.exe'
    analysis_exe_name = r'..\install\bin\analysis.exe'
    cv_script = r'CV.py'
else:
    SOL_exe_name = '../install/bin/SOL'
    vw_exe_name = 'vw'
    liblinar_train_exe_name = '../extern/liblinear/train'
    liblinar_test_exe_name = '../extern/liblinear/test'
    analysis_exe_name = '../install/bin/analysis'
    cv_script = r'./CV.py'
