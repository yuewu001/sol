================================================
SOL - A Large Scale Sparse Online Learning Library
================================================ 

                    V0.2.0                    

Available at 

About SOL
===========================================================================
SOL is an open-source library for large-scale sparse online learning, which
consists of a family of efficient and scalable sparse online learning
algorithms for large-scale online  classification tasks. We have offered
easy-to-use command-line tools and examples for users and developers. We also
have made comprehensive documents available for both beginners and advanced
users. SOL is not only a machine learning tool, but also a comprehensive
experimental platform for conducting large scale sparse online learning
research.

Specifically, SOL consists of a family of first order sparse online learning algorithms as follows
- STG: sparse online learning via truncated graidient (Langford et al., 2009);
- FOBOS: Forward backward splitting (Duchi et al., 2009);
- RDA: Regularized dual averaging(Xiao, 2010);
- RDA_E: Regularized dual averaging(Xiao, 2010);
,a family of second order sparse online learning algorithms as follows
- Ada-FOBOS : adaptive FOBOS (Duchi et al., 2011);
- Ada-RDA: adaptive RDA(Duchi et al., 2011);
- AROW:  adaptive regularization of weight vectors (Crammer et al., 2009);
- AROW-TG: adaptive regularization of weight vectors  with trunated gradient
- AROW-DA: adaptive regularization of weight vectors  with dual averaging
and a family of online feature selection algorithms:
- PET: Perceptron with truncation
- FOFS: first order online feature selection (Jialei et al. 2012) 
- SOFS: second online feature selection with adaptive regularization of weight vectors  for

This document briefly explains the usage of SOL. A more detailed manual can be
found from tutorail of SOL.

To get started, please read the ``Quick Start'' section first.

Table of Contents
=================
- Structure of folders
- Installation
- Quick Start
- Additional Information

Structure of folders
======================
---data 
    example datasets
---doc
    documentation of the library
---exp_sol
    python and matlab scripts for l1-regularized sparse online learning experiments, including cross validation and performance evaluation
---exp_ofs
    python and matlab scripts for online feature selection experiments, including cross validation and performance evaluation
---src
    source code of the library
---test
    est code and example use of the library
---tools
    python scripts to pre-process datasets
---CMakeLists.txt

Installation
======================
SOL features a very simple installation procedure. The project is managed by Cmake. There exists a `CMakeLists.txt` in the root dir of SOL. 

1. For linux users
    
    1. `cd` to the directory of SOL

    2. make a folder for building the project, like  `mkdir build`

    3. `cd` to the folder above and call cmake 

    4. `make` and you will get an executable `SOL` in the `bin` folder

    5. `make install` and the executable will be copied to the root dir of SOL

2. For windows users
    
    1. make a folder for building the project
    
    2. call cmake. Remember to specify the visual studio. For example, if you are using Visual Studio 2012, you can
       generate the project by

            cmake -G "Visual Studio 11" ..

    3. Open the project, Rebuild the `ALL_BUILD` project and then build the `INSTALL` project

3. For linux users with Eclipse

    1. `cd` to the directory of SOL
    
    2.  make a folder for building the project, like  `mkdir build`

    3. `cd` to the folder above and generate the project by

            cmake -G"Eclipse CDT4 - Unix Makefiles" -D CMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=/usr/local ..

    4. Install cmakeed plugin into Eclipse.
    	
    	     cmakeed: http://cmakeed.sourceforge.net/updates/

    5. Import the existing project from `build` into Eclipse.
    
    6. Build the project throught right click on the project, and then select 'Make Targets' -> 'Build'.

Quick Start
===========
Running SOL without any arguments or with '--help' will produce a message which briefly explains each argument. Below
arguments are grouped according to their function.

We provide an example to show how to use SOL and explain the details of how SOL works.
The dataset we use will be `a6a`. Note that only LibSVM datasets are supported by default.

The command for training wit default algorithm is as the following shows.
    
    ./SOL -i a6a -opt SGD

In this part, we will explain how to induce sparsity of the weight vector and how to tune parameters of algorithms.
In STG, we can induce sparsity by:
    
    ./SOL -i a6a -opt STG -eta 0.01 -l1 1e-3
    
Also,we can change the number of steps to truncate the gradients (default is 10).

    ./SOL -i a6a -opt STG -eta 0.01 -l1 1e-3 -k 1

For  details, please check tutorial of SOL.

Additional Information
======================

For any questions and comments, please send your email to
chhoi@ntu.edu.sg

Released date: 1 March, 2014.
