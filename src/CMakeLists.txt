cmake_minimum_required(VERSION 2.8)

set (solm_files
    src/algorithms/om/olm/solm/Ada_FOBOS.h
    src/algorithms/om/olm/solm/Ada_RDA.h
    src/algorithms/om/olm/solm/CW_RDA.h
    src/algorithms/om/olm/solm/CW_TG.h
    src/algorithms/om/olm/solm/DAROW.h
    src/algorithms/om/olm/solm/FOBOS.h
    src/algorithms/om/olm/solm/mRMR_OGD.h
    src/algorithms/om/olm/solm/RDA.h
    src/algorithms/om/olm/solm/STG.h
    src/algorithms/om/olm/solm/SparseOnlineLinearModel.h
    src/algorithms/om/olm/solm/solm_header.h
    PARENT_SCOPE
    )

set (ofs_files
    src/algorithms/om/olm/ofs/PET.h
    src/algorithms/om/olm/ofs/FOFS.h
    src/algorithms/om/olm/ofs/SOFS.h
    src/algorithms/om/olm/ofs/OnlineFeatureSelection.h
    src/algorithms/om/olm/ofs/ofs_header.h
    PARENT_SCOPE
    )

set (olm_files
    src/algorithms/om/olm/olm_header.h
    src/algorithms/om/olm/OnlineLinearModel.h
    src/algorithms/om/olm/SGD.h
    PARENT_SCOPE
    )

set (om_files
    src/algorithms/om/OnlineModel.h
    src/algorithms/om/om_header.h
    PARENT_SCOPE
    )

set (algo_files
    src/algorithms/algo_header.h
    src/algorithms/LearnModel.h
    PARENT_SCOPE
    )


set (loss_files
    src/loss/HingeLoss.h
    src/loss/LogisticLoss.h
    src/loss/loss_header.h
    src/loss/LossFunction.h
    src/loss/SquaredHingeLoss.h
    src/loss/SquareLoss.h
    PARENT_SCOPE
    )

set (opti_files
    src/optimizers/OnlineOptimizer.h
    src/optimizers/opt_header.h
    src/optimizers/Optimizer.h
    PARENT_SCOPE
    )

set(util_files 
    src/utils/config.h
    src/utils/ezOptionParser.hpp
    src/utils/init_param.h
    src/utils/MaxHeap.h
    src/utils/md5.h
    src/utils/MinHeap.h
    src/utils/Params.cpp
    src/utils/Params.h
    src/utils/reflector.h
    src/utils/s_array.h
    src/utils/thread_primitive.h
    src/utils/util.h
    src/utils/error.h
    PARENT_SCOPE
    )

set(converter_files 
    src/converter/Converter.cpp
    src/converter/Converter.h
    src/converter/Params.cpp
    src/converter/Params.h
    PARENT_SCOPE
    )

set  (io_files
    src/io/BatchDataSet.h
    src/io/DataChunk.h
    src/io/DataHandler.h
    src/io/DataPoint.h
    src/io/DataReader.h
    src/io/DataSet.h
    src/io/MPChunk.h
    src/io/OnlineBuffer.h
    src/io/OnlineDataSet.h
    src/io/OnlineDataSetHelper.h
    src/io/OnlineMPBuffer.h
    src/io/basic_io.cpp
    src/io/basic_io.h
    src/io/binary_io.h
    src/io/comp.h
    src/io/csv_io.h
    src/io/io_header.h
    src/io/io_interface.h
    src/io/libsvm_io.h
    src/io/parser.h
    PARENT_SCOPE
    )

set (converter_files
    src/converter/ConverterParams.h
    src/converter/ConverterParams.cpp
    src/converter/Converter.h
    src/converter/Converter.cpp
    PARENT_SCOPE
    )
