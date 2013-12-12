# Documentation
SOL is an large scale sparse online learning library, which consists of a family of 
efficient and scalable state-of-art online learning algorithms for large-scale sparse 
online classification tasks. The library provide an easy-to-use command-line tool and 
several scripts for users and developers.  We performed comprehensive experiments to 
verify the efficiency and effectiveness of the library.

##Outline
1. [Structure of folders](#struct)

2. [Command line](#cmd_line)

3. [Tutorial](#tutorial)

    3.1. [Installation](#install)

    3.2. [A step by step example](#step_example)

4. [User Manual & Extend the library](#manual)

    4.1. [IO Handlers](#io_handler)

    <ul style="list-style-type:none;">
        <li><p><a href="#parse">4.1.1. Parsing Dataset</a></p></li>
        <li><p><a href="#extend_datareader">4.1.2. Extend the data readers</a></p></li>
    </ul>
        
    4.2. [Loss Functions](#loss)

    4.3. [Optimizers](#optimizer)

     <ul style="list-style-type:none;">
        <li><p><a href="#opt_detail">4.3.1. Details of Optimizer</a></p></li>
        <li><p><a href="#opt_own">4.3.2. Implement your own algorithms</a></p></li>
    </ul>
   
    4.4. [Common Utilities](#common_util)

<!--
5. [Experimental Results of SOL](#exp)

    5.1. [Experiment on RCV1](#exp_rcv1)

    5.2. [Experiment on kdda](#exp_kdda)
-->


## <a id='struct'>1. Structure of folders</a>
*   ---doc

    documentation of the library

*   ---exp

    some scripts for experiments

*   ---src

    -   ---common

        source code of common utilities and global definitions

    -   ---data

        source code of IO handlers

    -   ---loss

        source code of loss functions

    -   ---optimizer

        source code of large scale sparse online learning algorithms

*   ---tools

    Some tools to standardize datasets


## <a id='cmd_line'>2. Command line</a>

Running SOL without any arguments or with '--help' will produce a message which briefly explains each argument. Below
arguments are grouped according to their function.
#### Input Options
    -i  arg :    file path to the training data
    -c  arg :    file path to the cached training data
    -t  arg :    file path to the test data
    -tc arg :    file path to the cached test data
    -dt arg :    data type option, only LibSVM is supported by default
    -bs arg :    number of chunks for buffering, default is 2

### Loss Functions
    -loss arg:      loss function type

    supported loss functions:
    *   Hinge:          hinge loss 
    *   Logit:          logistic loss
    *   Square:         square loss
    *   SquareHinge:    squared hinge loss

### Algorithms and parameters
    -opt arg :      optimization algorithms, options include:
                    SGD, STG, RDA, RDA_E, FOBOS, Ada-RDA, Ada-FOBOS, AROW
    -eta arg :      learning rate, if this option is not specified, algorithms will 
                    try difference parameters and select a best one automatically         
    -l1  arg :      L1 regularization parameter
    -passes arg:    number of passes to go through 

#### Algorithm list
*   **SGD**

    Stochastic Gradient Descent algorithm. Note that no sparsity is induced even a L1 regularization parameter is set.

*   **STG**

    Truncated Gradient Descent algorithm. 

    *Parameter*:

        -k  arg :       truncate the weight vector every k steps

    *Reference*: 

        Langford J, Li L, Zhang T. Sparse online learning via truncated gradient[J]. The Journal of Machine Learning Research, 2009, 10: 777-801. 

*   **RDA**

    Regularized Dual Averaging algorithm. L1 regularization is implemented.

    *Reference*:

        Xiao L. Dual averaging methods for regularized stochastic learning and online optimization[J]. The Journal of Machine Learning Research, 2010, 9999: 2543-2596.  

*   **RDA_E**

    Enhanced L1-Regularized Dual Averaging algorithm.

    *Parameter*: 

        -gammarou arg :      decreased L1 regularization parameter.

    *Reference*:

        Xiao L. Dual averaging methods for regularized stochastic learning and online optimization[J]. The Journal of Machine Learning Research, 2010, 9999: 2543-2596.  

*   **FOBOS**

    Forward Backward Splitting algorithm.

    *Reference*:

        Duchi J, Singer Y. Efficient online and batch learning using forward backward splitting[J]. The Journal of Machine Learning Research, 2009, 10: 2899-2934.  

*   **Ada-RDA**

    Adaptive Regularized Dual Averaging algorithm.

    *Parameter*:

        -delta arg :    parameter to ensure positive-definite property of the adaptive weighting matrix

    *Reference*:

        Duchi J, Hazan E, Singer Y. Adaptive subgradient methods for online learning and stochastic optimization[J]. The Journal of Machine Learning Research, 2011, 999999: 2121-2159.

*   **Ada-FOBOS**

    Adaptive Forward Backward Splitting algorithm.

    *Parameter*:

        -delta arg :    parameter to ensure positive-definite property of the adaptive weighting matrix

    *Reference*:

        Duchi J, Hazan E, Singer Y. Adaptive subgradient methods for online learning and stochastic optimization[J]. The Journal of Machine Learning Research, 2011, 999999: 2121-2159.

*   **AROW**

    Adaptive Regularization of weighted vectors.

    *Parameter*:

        -r arg :    parameter of passive-aggressive update trade-off

    *Reference*:

        rammer K, Kulesza A, Dredze M. Adaptive regularization of weight vectors[J]. Machine Learning, 2009: 1-33.

*   **ASAROW**

    Adaptive Regularization of weighted vectors for feature selection.

    *Parameter*:

        -k arg :    number of weight dimenstions to keep

*   **SSAROW**

    Truncated Sparse Adaptive Regularization of weighted vectors.

    *Parameter*:

        -r arg :    parameter of passive-aggressive update trade-off

*   **CW-RDA**

    Confidence weighted dual avearaging regularization

    *Parameter*:

        -r arg :    parameter of passive-aggressive update trade-off

*   **SCW**

    Exact Soft-Confidence Weighted Learning (To be verified)

    *Parameter*:

        -phi arg:   probability parameter in SCW

        -r arg  :    parameter of passive-aggressive update trade-off

*   **SCW-RDA**

    Exact Soft-Confidence Weighted Learning with regularized dual averaing (To be verified)

    *Parameter*:

        -phi arg:   probability parameter in SCW

        -r arg  :    parameter of passive-aggressive update trade-off

## <a id='tutorial'>3. Tutorial</a>

### <a id='install'>3.1 Installation</a>
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

In the `vs` directory, we provide a Visual Studio 2012 Project. 


### 3.2 <a id='step_example'>A step-by-step example</a>
In this section, we provide an example to show how to use SOL and explain the details of how SOL works.
The dataset we use will be `a6a`. Note that only LibSVM datasets are supported by default.

#### 3.2.1 SGD
The command for training is the following.
    
    ./SOL -i a6a -opt SGD

The output will be: 
    
    eta0 = 1e-08    mistake rate: 24.0107 %
    eta0 = 1e-07    mistake rate: 24.0107 %
    eta0 = 1e-06    mistake rate: 24.0107 %
    eta0 = 1e-05    mistake rate: 24.0107 %
    eta0 = 0.0001   mistake rate: 24.0107 %
    eta0 = 0.001    mistake rate: 20.8913 %
    eta0 = 0.01 mistake rate: 17.041 %
    eta0 = 0.1  mistake rate: 17.3797 %
    eta0 = 1    mistake rate: 20.6506 %
    Best Parameter: eta = 0.01
    
    --------------------------------------------------
    Algorithm: STG
    Learn error rate: 17.04 +/- 0.00 %
    Sparsification Rate: 0.00 %
    Learning time: 0.110 s

First, SOL will try different learning rates and then select the best one. If you do know the best learning rate,
you can specify by:

    ./SOL -i a6a -opt SGD -eta 0.01


#### 3.2.2 STG
In this part, we will explain how to induce sparsity of the weight vector and how to tune parameters of algorithms.
In STG, we can induce sparsity by:
    
    ./SOL -i a6a -opt STG -eta 0.01 -l1 1e-3
    
The output will be:

    --------------------------------------------------
    Algorithm: STG
    Learn error rate: 17.17 +/- 0.00 %
    Sparsification Rate: 36.07 %
    Learning time: 0.027 s

We can see that the sparsification rate is much larger than that of SGD.

Also,we can change the number of steps to truncate the gradients (default is 10).

    ./SOL -i a6a -opt STG -eta 0.01 -l1 1e-3 -k 1

The output is almost the same to the default in this example.


## 4. <a id='manual'>User Manual & Extend the library</a>

In this section, we will explain the details of the source code. The library is constituted of four major parts:
*IO Handler*, *Loss Functions*, *Optimizers*, and *Common Utilities*.

### 4.1 <a id='io_handler'>IO Handler</a>
IO Handler is in charge of data loading. The three major functions are: parsing the original dataset, caching data, and
a common interface with optimizers.

#### 4.1.1 <a id='parse'>Parsing dataset</a>
It requires different codes to parse different formats of data. By default, we only support LibSVM format. The base
class to parse a dataset is `DataReader` (in *DataReader.h*), in which we define the interfaces to load a dataset file
correctly. The interfaces are:

*   OpenReading
    
        virtual bool OpenReading() = 0;

    Open a dataset file to load data. Note that we do not specify the source of dataset (like a file path name), as the 
    source of data may be online sources (like TCP). The `open` operation can go beyond opening a local file. It can
    also open a socket listener.

    Return true if everything is ok.

*   GetNextData

        virtual bool GetNextData(DataPoint<FeatType, LabelType> &data) = 0;

    Get a new data point from the source. `data` is the variable to place the obtained data.

    Return true if everything is ok.

*   Rewind

        virtual void Rewind() = 0;

    Rewind the data source to the beginning

*   Close

        virtual void Close() = 0;

    Close the data source when loading is finished.

*   Good
    
        virtual bool Good() = 0;

    Test if the data reader is ok.

#### 4.1.2 <a id='extend_datareader'>Extend the data readers</a>

For a specific format of data, we only need to inherit from the  `DataReader` class and implement the above interfaces.
It will work when you assign the customized data reader to the dataset.

The file `libsvmread.h` can be regarded as an example to extend the `DataReader`.

### 4.2 <a id='loss'>Loss Functions</a>
At the moment, we provide a base class (purely virtual class) for loss functions and two child classes(HingleLoss and
Logistic Loss). The interfaces are:

*   IsCorrect

        virtual inline bool IsCorrect(LabelType label, float predict);

    This function is implemented in the base class to justify whether a prediction is correct for binary classification
    problems. We assign the virtual property to it for the extensibility to multi-class or regression problems.

*   GetLoss

        virtual float GetLoss(LabelType label, float predict) = 0;

    Get the loss of the current prediction

*   GetGradient

        virtual float GetGradient(LabelType label, float predict) = 0;

    Get the gradient of the loss function at the current data point. Note that we do not calculate the exact gradient
    here. To linear classification problems, the gradients on different features share a same part. Take Hinge Loss for
    example:

    <i>l(<b>w</b>) = 1 - y <b>w</b> &#8226; <b>x</b></i>

    The gradient is:

    <i>l&#39;(<b>w</b>) = -y <b>x</b></i>

    As a result, we only calculate the shared term `-y` for the gradients of different features for efficiency concern. 
    Users need to multiply the correspondent feature `x[i]` in the optimization algorithms.

#### 4.2.1 Extend loss functions
The files 'HingeLoss.h', 'LogisticLoss.h', 'SquareLoss.h', and
'SquareHingeLoss.h'  are three examples to extend the base class.

### 4.3 <a id='optimizer'>Optimizers</a>

Optimizers are the online learning algorithms.  The base class `Optimizer` implements the details of how a linear
classification model works, including interacting with a dataset, training the model, updating the model, test the
model, and some other auxiliary functions. It also define the interfaces for different learning algorithms(those virtual
functions).

#### 4.3.1 <a id='opt_detail'>Details of Optimizer class</a>

##### Class Members

*   `curIterNum`:   current iteration number
*   `initial_t`:    initial iteration number, this is the variable to avoid large learning rates at the beginning
*   `eta0`:         initial learning rate
*   `eta` :         learning rates. This variable is set to `eta0` at the when training begins. Different algorithms can
                    set the value of learning rate on the fly.
*   `lambda`:       The L1 regularization parameter.
*   `dataSet`:      The training and test data source.   
*   `weightVec`:    The weight vector of the linear model.
*   `weightDim`:    current dimension of the weight vector. Note that this variable is changing with training data
                    coming in.
*   `sparse_soft_thresh`: 
                    threshold below which a weight is regarded as zero.
*   `lossFunc` :    user specified loss function
*   `id_str` :      a string to describe the algorithm, need to be assigned a value when an algorithm is constructed


##### Class Methods

**Constructor**

        Optimizer(DataSet<FeatType, LabelType> &dataSet, LossFunc<FeatType, LabelType> &lossFunc); 

When initializing an optimizer, users need to assign the data set and the loss function at least.

**Destructor**

        virtual ~Optimizer();

Destroy the optimizer and release memory.

**Learning**

1.  Learn

        float Learn(int numOfTimes = 1);

    Learn a model and return the average error rate. Note that the input parameter is only used for those dataset that
    can be randomized. It is not available at the moment.

        float Learn(float &aveErrRate, float &varErrRate, float& sparseRate, int numOfTimes = 1);

    Learn a model and return the average error rate.

    **Parameters**
    *   `aveErrRate`:   average error rate
    *   `varErrRate`:   variance of the average error rate, only valid when dataset can be randomized
    *   `sparesRate`:   sparsification rate of the linear model
    *   `numOfTimes`:   number of times to learn the model with randomized dataset, not available at the moment

2.  BeginTrain
    
        virtual void BeginTrain();

    Reset the optimizer to the initialization status  of training

    **Note:**   Users should call this base function explicitly in their inherited function to ensure the model is reset correctly.

3.  Train

        float Train();

    Train the model and return the learning error rate.

4.  EndTrain

        virtual void EndTrain();

    Called when training is finished. 

5.  Predict

        float Predict(DataPoint<FeatType, LabelType> &data);

    Predict the label of the input data point.


6.  UpdateWeightVec

        virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x) = 0;

    The core function of learning. This function is called each time a new data point comes to update the model.

    `x`: the new data that comes in

    **Return**: the prediction of the input data `x`

**Test**

        float Test(DataSet<FeatType ,LabelType> &testSet);

Test the performance of the given dataset. Return the test error rate.

**Auxiliary Function**

1.  SetParameter
    
        void SetParameter(float lambda = -1, float eta0 = -1);

    Set the learning rate and L1 regularization parameter. `-1` means no change.

2.  BestParameter()

        virtual void BestParameter();

    Learn the best learning rate by default. It can be inherited and learn other parameters to satisfy the requirements
    of different algorithms.

3.  GetSparseRate
    
        float GetSparseRate(int total_len = 0);

    Get the sparse rate of the model. The total_len is the dimension of the input data. If not assigned by users, the 
    largest index of features will be used.

4.  UpdateWeightSize

        virtual void UpdateWeigthSize(int newDim);

    Update the dimension of the weight vector. As we are learning on sparse data online, we do not know the dimension of
    the input data. So the weight vector needs to be resized on the fly. 
    Note that inherited algorithms need to override this function to resize their own dimension-related members and call 
    the base one explicitly to resize the weight vector. 

5.  PrintOptInfo
    
        void PrintOptInfo() const;

    Print the optimization information.

6.  Id_Str

        const string& Id_Str() const;

    Get the identity string of the optimizer.

#### 4.3.2 <a id='opt_own'>Implement your own algorithms</a>

To implement a specific learning algorithm, you only need to inherit from the base class `Optimizer`, and implement
the pure virtual function `UpdateWeightVec`. Whether other virtual functions need to be override depends on the
specific algorithm. Take STG for example, it has to maintain a time stamp vector. So it override the `BeginTrain`
and `UpdateWeightSize` functions to initialize and resize the time stamp vector. It needs to shrink weight vectors
at the end of the training. So `EndTrain` is override. 

Check the implemented algorithms to explore more details of how to extend the optimizers.

### 4.4 <a id='common_util'>Common Utilities</a>

For some utilized functions and global definitions, please refer to
*'common/init_param.h'* and *'common/util.h'*.


<br/>
<br/>
<!--
## <a id='exp'>Experimental Results of SOL</a>

### <a id='exp_rcv1'>Experiment on RCV1 </a>

### <a id='exp_kdda'>Experiment on kdda</a>
-->
