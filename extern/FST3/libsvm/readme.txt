Feature Selection Toolbox 3 optionally takes use of the
LibSVM library to enable SVM based wrappers (i.e., feature 
selection directly optimizing SVM classification accuracy)

To enable SVM support in FST3 place into this directory
the files svm.cpp and svm.h, that can be downloaded from:

http://www.csie.ntu.edu.tw/~cjlin/libsvm/

Remark 1: On some systems LibSVM comes pre-installed.
In such case to take use of the pre-installed library
please adjust Makefile manually (path to LibSVM sources
is defined in LIBSVMDIR, binaries should be added to INCLUDE).

Remark 2: LibSVM by default prints detailed logs in the
course of classifier training. Together with FST3 messages
this may lead to cluttered output. Consider disabling
the function static void info(const char *fmt,...) in
svm.cpp by changing the respective
#if 1 
directive in front of static void info(const char *fmt,...) to 
#if 0
(valid at least in versions 2.91 and 3.00 of LibSVM)
