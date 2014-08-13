/* =========================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
*/ /*! 

   \file    demo53.cpp 
   \brief   Example 53: Hybrid feature selection
   \example demo53.cpp 
            Implements \ref example53, see also 
            <a href="http://fst.utia.cz/?fst3_usage#example53">Example 53 source code</a>
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see 
            <a href="http://fst.utia.cz/?contacts">Contacts</a>
   \date    March 2011
   \version 3.1.0.beta
   \note    FST3 was developed using gcc 4.3 and requires
   \note    \li Boost library (http://www.boost.org/, tested with versions 1.33.1 and 1.44)
   \note    \li (\e optionally) LibSVM (http://www.csie.ntu.edu.tw/~cjlin/libsvm/, 
                tested with version 3.00)
   \note    Note that LibSVM is required for SVM related tools only,
            as demonstrated in demo12t.cpp, demo23.cpp, demo25t.cpp, demo32t.cpp, etc.

*/ /* 
=========================================================================
Copyright:
  * FST3 software (with exception of any externally linked libraries) 
    is copyrighted by Institute of Information Theory and Automation (UTIA), 
    Academy of Sciences of the Czech Republic.
  * FST3 source codes as presented here do not contain code of third parties. 
    FST3 may need linkage to external libraries to exploit its functionality
    in full. For details on obtaining and possible usage restrictions 
    of external libraries follow their original sources (referenced from
    FST3 documentation wherever applicable).
  * FST3 software is available free of charge for non-commercial use. 
    Please address all inquires concerning possible commercial use 
    of FST3, or if in doubt, to FST3 maintainer (see http://fst.utia.cz)
  * Derivative works based on FST3 are permitted as long as they remain
    non-commercial only.
  * Re-distribution of FST3 software is not allowed without explicit
    consent of the copyright holder.
Disclaimer of Warranty:
  * FST3 software is presented "as is", without warranty of any kind, 
    either expressed or implied, including, but not limited to, the implied 
    warranties of merchantability and fitness for a particular purpose. 
    The entire risk as to the quality and performance of the program 
    is with you. Should the program prove defective, you assume the cost 
    of all necessary servicing, repair or correction.
Limitation of Liability:
  * The copyright holder will in no event be liable to you for damages, 
    including any general, special, incidental or consequential damages 
    arising out of the use or inability to use the code (including but not 
    limited to loss of data or data being rendered inaccurate or losses 
    sustained by you or third parties or a failure of the program to operate 
    with any other programs).
========================================================================== */

#include <boost/smart_ptr.hpp>
#include <exception>
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include "error.hpp"
#include "global.hpp"
#include "subset.hpp"

#include "data_intervaller.hpp"
#include "data_splitter.hpp"
//#include "data_splitter_5050.hpp"
#include "data_splitter_cv.hpp"
//#include "data_splitter_holdout.hpp"
//#include "data_splitter_leave1out.hpp"
//#include "data_splitter_resub.hpp"
#include "data_splitter_randrand.hpp"
//#include "data_splitter_randfix.hpp"
#include "data_scaler.hpp"
//#include "data_scaler_void.hpp"
#include "data_scaler_to01.hpp"
//#include "data_scaler_white.hpp"
#include "data_accessor_splitting_memTRN.hpp"
#include "data_accessor_splitting_memARFF.hpp"

#include "criterion_normal_bhattacharyya.hpp"
//#include "criterion_normal_gmahalanobis.hpp"
//#include "criterion_normal_divergence.hpp"
//#include "criterion_multinom_bhattacharyya.hpp"
#include "criterion_wrapper.hpp"
//#include "criterion_wrapper_bias_estimate.hpp"
//#include "criterion_subsetsize.hpp"
//#include "criterion_sumofweights.hpp"
//#include "criterion_negative.hpp"

//#include "distance_euclid.hpp"
//#include "distance_L1.hpp"
//#include "distance_Lp.hpp"
//#include "classifier_knn.hpp"
//#include "classifier_normal_bayes.hpp"
//#include "classifier_multinom_naivebayes.hpp"
#include "classifier_svm.hpp"

//#include "search_bif.hpp"
//#include "search_bif_threaded.hpp"
//#include "search_monte_carlo.hpp"
//#include "search_monte_carlo_threaded.hpp"
//#include "search_exhaustive.hpp"
//#include "search_exhaustive_threaded.hpp"
//#include "branch_and_bound_predictor_averaging.hpp"
//#include "search_branch_and_bound_basic.hpp"
//#include "search_branch_and_bound_improved.hpp"
//#include "search_branch_and_bound_partial_prediction.hpp"
//#include "search_branch_and_bound_fast.hpp"
//#include "seq_step_straight.hpp"
//#include "seq_step_straight_threaded.hpp"
#include "seq_step_hybrid.hpp"
//#include "seq_step_ensemble.hpp"
#include "search_seq_sfs.hpp"
//#include "search_seq_sffs.hpp"
//#include "search_seq_sfrs.hpp"
//#include "search_seq_os.hpp"
//#include "search_seq_dos.hpp"
//#include "result_tracker_dupless.hpp"
//#include "result_tracker_regularizer.hpp"
//#include "result_tracker_feature_stats.hpp"
//#include "result_tracker_stabileval.hpp"


/*! \section example53  Example 53: Hybrid feature selection. 

Selects features using hybridized SBS algorithm where the primary criterion is 
SVM wrapper estimated classification accuracy and the pre-filtering criterion is 
normal Bhattacharyya distance. In each algorithm step all current feature subset 
candidates are evaluated by means of Bhattacharyya, 50% of the worst is abandoned 
and only the remaining 50% are evaluated by means of the slow SVM wrapper to select 
intermediate solutions. 50% of data is randomly chosen to form the training dataset 
(remains the same for all the time), 40% of data is randomly chosen to be used at 
the end for validating the classification performance on the finally selected
subspace. (selected training and test data parts are disjunct and altogether
cover 90% of the original data). SBS is called in d-optimizing setting, 
invoked by parameter 0 in search(0,...), which is otherwise
used to specify the required subset size.
*/
int main()
{
	try{
	typedef double RETURNTYPE; 	typedef double DATATYPE;  typedef double REALTYPE;
	typedef unsigned int IDXTYPE;  typedef unsigned int DIMTYPE;  typedef short BINTYPE;
	typedef FST::Subset<BINTYPE, DIMTYPE> SUBSET;
	typedef FST::Data_Intervaller<std::vector<FST::Data_Interval<IDXTYPE> >,IDXTYPE> INTERVALLER;
	typedef boost::shared_ptr<FST::Data_Splitter<INTERVALLER,IDXTYPE> > PSPLITTER;
	typedef FST::Data_Splitter_CV<INTERVALLER,IDXTYPE> SPLITTERCV;
	typedef FST::Data_Splitter_RandomRandom<INTERVALLER,IDXTYPE,BINTYPE> SPLITTERRR;
	typedef FST::Data_Accessor_Splitting_MemARFF<DATATYPE,IDXTYPE,INTERVALLER> DATAACCESSOR; // uncomment for TRN data format
	//typedef FST::Data_Accessor_Splitting_MemARFF<DATATYPE,IDXTYPE,INTERVALLER> DATAACCESSOR; // uncomment for ARFF data format
	typedef FST::Criterion_Normal_Bhattacharyya<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> BHATTCRIT;
	typedef FST::Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> CLASSIFIERSVM;
	typedef FST::Criterion_Wrapper<RETURNTYPE,SUBSET,CLASSIFIERSVM,DATAACCESSOR> WRAPPERSVM;
	typedef FST::Sequential_Step_Hybrid<RETURNTYPE,DIMTYPE,SUBSET,WRAPPERSVM,BHATTCRIT> EVALUATOR;

	string data_file = "D:/Coding/NTU/sol/tools/a9a.arff2";
		std::cout << "Starting Example 53: Hybrid feature selection..." << std::endl;
	// randomly sample 50% of data for training and randomly sample (disjunct) 40% for independent testing of final classification performance 
		PSPLITTER dsp_outer(new SPLITTERRR(1, 50, 40)); // (there will be one outer randomized split only)
	// in the course of search use the first half of data by 3-fold cross-validation in wrapper FS criterion evaluation
		PSPLITTER dsp_inner(new SPLITTERCV(3));
	// scale data to [0,1] by simple histogram shrinking
		boost::shared_ptr<FST::Data_Scaler<DATATYPE> > dsc(new FST::Data_Scaler_to01<DATATYPE>());
	// set-up data access
		boost::shared_ptr<std::vector<PSPLITTER> > splitters(new std::vector<PSPLITTER>); 
		splitters->push_back(dsp_outer); splitters->push_back(dsp_inner);
		boost::shared_ptr<DATAACCESSOR> da(new DATAACCESSOR(data_file,splitters,dsc));
		da->initialize();
	// initiate access to outer split data parts
		da->setSplittingDepth(0); if(!da->getFirstSplit()) throw FST::fst_error("50/40 random data split failed.");
	// set-up normal Bhattacharyya criterion to serve as pre-filtering criterion in hybrid search
	// (note that it is to be initialized in outer splitter context (i.e., in splitting depth 0, meaning on all training data))
		boost::shared_ptr<BHATTCRIT> cb(new BHATTCRIT);
		cb->initialize(da);
	// initiate access to inner split data parts
	// (note that wrapper evaluation in the course of search works in inner splitting loop (for 3-fold cross-validated SVM accuracy estimation))
		da->setSplittingDepth(1); if(!da->getFirstSplit()) throw FST::fst_error("3-fold cross-validation failure.");
	// initiate the storage for subset to-be-selected
		boost::shared_ptr<SUBSET> sub(new SUBSET(da->getNoOfFeatures()));
	// set-up SVM (interface to external library LibSVM)
		boost::shared_ptr<CLASSIFIERSVM> csvm(new CLASSIFIERSVM);
		csvm->initialize(da);
	// wrap the SVM classifier to enable its usage as FS criterion (criterion value will be estimated by 3-fold cross-val.)
		boost::shared_ptr<WRAPPERSVM> wsvm(new WRAPPERSVM);
		wsvm->initialize(csvm,da);
	// set-up the hybrid search step object so that 50% of subset candidates is pre-filtered in each step (option: straight, ensemble, etc.)
		boost::shared_ptr<EVALUATOR> eval(new EVALUATOR(cb,50/*_keep_perc*/));
	// set-up Sequential Backward Selection search procedure
		FST::Search_SFS<RETURNTYPE,DIMTYPE,SUBSET,WRAPPERSVM,EVALUATOR> srch(eval);
		srch.set_search_direction(FST::BACKWARD);
	// run the search
		std::cout << "Feature selection setup:" << std::endl << *da << std::endl << srch << std::endl << *wsvm << std::endl << std::endl;
		RETURNTYPE critval_train, critval_test;
		srch.set_output_detail(FST::NORMAL); // set FST::SILENT to disable all text output in the course of search (FST::NORMAL is default)
		if(!srch.search(50,critval_train,sub,wsvm,std::cout)) throw FST::fst_error("Search not finished.");
	// (optionally) validate result by estimating SVM accuracy on selected feature sub-space on independent test data
		da->setSplittingDepth(0);
		csvm->train(da,sub);
		csvm->test(critval_test,da);
		std::cout << "Validated SVM accuracy=" << critval_test << std::endl << std::endl;
	}
	catch(FST::fst_error &e) {std::cerr<<"FST ERROR: "<< e.what() << ", code=" << e.code() << std::endl;}
	catch(std::exception &e) {std::cerr<<"non-FST ERROR: "<< e.what() << std::endl;}
	return 0;
}
