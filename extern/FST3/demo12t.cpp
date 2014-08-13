/* =========================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
*/ /*! 

   \file    demo12t.cpp 
   \brief   Example 12t: Threaded SVM-wrapper-based feature selection with Dynamic Oscillating Search
   \example demo12t.cpp 
            Implements \ref example12t, see also 
            <a href="http://fst.utia.cz/?fst3_usage#example12t">Example 12t source code</a>
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
#include "data_splitter_5050.hpp"
#include "data_splitter_cv.hpp"
//#include "data_splitter_holdout.hpp"
//#include "data_splitter_leave1out.hpp"
//#include "data_splitter_resub.hpp"
//#include "data_splitter_randrand.hpp"
//#include "data_splitter_randfix.hpp"
#include "data_scaler.hpp"
#include "data_scaler_void.hpp"
//#include "data_scaler_to01.hpp"
//#include "data_scaler_white.hpp"
#include "data_accessor_splitting_memTRN.hpp"
#include "data_accessor_splitting_memARFF.hpp"

//#include "criterion_normal_bhattacharyya.hpp"
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
#include "seq_step_straight_threaded.hpp"
//#include "seq_step_hybrid.hpp"
//#include "seq_step_ensemble.hpp"
//#include "search_seq_sfs.hpp"
//#include "search_seq_sffs.hpp"
//#include "search_seq_sfrs.hpp"
//#include "search_seq_os.hpp"
#include "search_seq_dos.hpp"
//#include "result_tracker_dupless.hpp"
//#include "result_tracker_regularizer.hpp"
//#include "result_tracker_feature_stats.hpp"
//#include "result_tracker_stabileval.hpp"


/*! \section example12t Example 12t: Threaded SVM-wrapper-based feature selection with Dynamic Oscillating Search.

Dynamic Oscillating Search is a d-optimizing procedure that adjusts selected
subset size in the course of search. It is a generalization of the Oscillating
Search idea, which proved to be useful in various feature selection contexts.
Here we demonstrate it in multi-threaded configuration (using \c Sequential_Step_Straight_Threaded
instead of \c Sequential_Step_Straight evaluator object). In this example
due to the use of very complex feature selection criterion (SVM Wrapper)
the speed gain due to multithreading is substantial. In this example features are selected 
on 40-dimensional \e waveform data with 3-fold cross-validated SVM wrapper 
as criterion on the first 50% of data samples. The final classification performance 
on the selected subspace is eventually validated on the second 50% of data.

\note The maximum permitted number of threads to run at once is 
to be user-specified with respect to hardware capabilities.
*/
int main()
{
	try{
	const unsigned int max_threads=2;
	typedef double RETURNTYPE; 	typedef double DATATYPE;  typedef double REALTYPE;
	typedef unsigned int IDXTYPE;  typedef unsigned int DIMTYPE;  typedef short BINTYPE;
	typedef FST::Subset<BINTYPE, DIMTYPE> SUBSET;
	typedef FST::Data_Intervaller<std::vector<FST::Data_Interval<IDXTYPE> >,IDXTYPE> INTERVALLER;
	typedef boost::shared_ptr<FST::Data_Splitter<INTERVALLER,IDXTYPE> > PSPLITTER;
	typedef FST::Data_Splitter_CV<INTERVALLER,IDXTYPE> SPLITTERCV;
	typedef FST::Data_Splitter_5050<INTERVALLER,IDXTYPE> SPLITTER5050;
	typedef FST::Data_Accessor_Splitting_MemTRN<DATATYPE,IDXTYPE,INTERVALLER> DATAACCESSOR;
	typedef FST::Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> CLASSIFIERSVM;
	typedef FST::Criterion_Wrapper<RETURNTYPE,SUBSET,CLASSIFIERSVM,DATAACCESSOR> WRAPPERSVM;
	typedef FST::Sequential_Step_Straight_Threaded<RETURNTYPE,DIMTYPE,SUBSET,WRAPPERSVM,max_threads> EVALUATOR;

		std::cout << std::endl << "Starting Example 12t: Threaded SVM-wrapper-based feature selection with Dynamic Oscillating Search..." << std::endl;
	// keep second half of data for independent testing of final classification performance
		PSPLITTER dsp_outer(new SPLITTER5050());
	// in the course of search use the first half of data by 3-fold cross-validation in wrapper FS criterion evaluation
		PSPLITTER dsp_inner(new SPLITTERCV(3));
	// do not scale data
		boost::shared_ptr<FST::Data_Scaler<DATATYPE> > dsc(new FST::Data_Scaler_void<DATATYPE>());
	// set-up data access
		boost::shared_ptr<std::vector<PSPLITTER> > splitters(new std::vector<PSPLITTER>); 
		splitters->push_back(dsp_outer); splitters->push_back(dsp_inner);
		boost::shared_ptr<DATAACCESSOR> da(new DATAACCESSOR("data/waveform_40.trn",splitters,dsc));
		da->initialize();
	// initiate access to split data parts
		da->setSplittingDepth(0); if(!da->getFirstSplit()) throw FST::fst_error("50/50 data split failed.");
		da->setSplittingDepth(1); if(!da->getFirstSplit()) throw FST::fst_error("3-fold cross-validation failure.");
	// initiate the storage for subset to-be-selected
		boost::shared_ptr<SUBSET> sub(new SUBSET(da->getNoOfFeatures()));
	// set-up SVM (interface to external library LibSVM)
		boost::shared_ptr<CLASSIFIERSVM> csvm(new CLASSIFIERSVM);
		csvm->set_kernel_type(RBF); // (option: LINEAR, POLY, SIGMOID)
		csvm->initialize(da);
	// wrap the SVM classifier to enable its usage as FS criterion (criterion value will be estimated by 3-fold cross-val.)
		boost::shared_ptr<WRAPPERSVM> wsvm(new WRAPPERSVM);
		wsvm->initialize(csvm,da);
	// set-up the threaded sequential search step object (option: hybrid, ensemble, etc.)
		boost::shared_ptr<EVALUATOR> eval(new EVALUATOR);
	// set-up Dynamic Oscillating Search procedure
		FST::Search_DOS<RETURNTYPE,DIMTYPE,SUBSET,WRAPPERSVM,EVALUATOR> srch(eval);
		srch.set_delta(3);
	// first optimize SVM parameters using 3-fold cross-validation on training data on the full set of features
		sub->select_all();
		csvm->optimize_parameters(da,sub);
	// run the search
		std::cout << "Feature selection setup:" << std::endl << *da << std::endl << srch << std::endl << *wsvm << std::endl << std::endl;
		sub->deselect_all(); // let DOS start from an empty set (any starting subset is permissible)
		RETURNTYPE critval_train, critval_test;
		if(!srch.search(0,critval_train,sub,wsvm,std::cout)) throw FST::fst_error("Search not finished.");
	// (optionally) the following line is included here just for illustration because srch.search() reports results in itself
		std::cout << std::endl << "Search result: " << std::endl << *sub << std::endl << "Criterion value=" << critval_train << std::endl << std::endl;
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
