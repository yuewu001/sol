/* =========================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
*/ /*! 

   \file    demo62.cpp 
   \brief   Example 62: (Missing data substitution) Combined feature subset contents, size and SVM parameters optimization
   \example demo62.cpp 
            Implements \ref example62, see also 
            <a href="http://fst.utia.cz/?fst3_usage#example62">Example 62 source code</a>
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
#include "seq_step_straight.hpp"
//#include "seq_step_straight_threaded.hpp"
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


/*! \section example62 Example 62: (Missing data substitution) Combined feature subset contents, size and SVM parameters optimization. 

In many tasks some of the values in data set are missing. Provided such
values are marked by a unique value, FST can handle such data -- in the
pre-processing phase each missing value is substituted by the average value
over all valid valued per feature.
This example processess incomplete data in the same way as in \ref example23 by means of
a repeated sequence of two consecutive operations - feature subset search followed by 
SVM parameter optimization for the current subset.
Features are selected using DOS algorithm and SVM (with sigmoid kernel) wrapper classification 
accuracy as FS criterion. 50% of data is randomly chosen to form the training 
dataset (remains the same for all the time), 40% of data is randomly chosen to be used 
at the end for validating the classification performance on the finally selected subspace 
(training and test data parts are disjunct and altogether cover 90% of the original data).
The training data part is accessed by means of 3-fold cross-validation in the course of 
search. The optimization process consists of repeated consecutive calls of two procedures: 
SVM parameter optimization followed by DOS feature subset optimization. (SVM parameters 
are optimized on the currently best known feature subset, which is then used to initialize 
next DOS search). The calls are repeated as long as better SVM performance (on the training 
data) is achieved.
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
	typedef FST::Data_Accessor_Splitting_MemTRN<DATATYPE,IDXTYPE,INTERVALLER> DATAACCESSOR; // uncomment for TRN data format
	//typedef FST::Data_Accessor_Splitting_MemARFF<DATATYPE,IDXTYPE,INTERVALLER> DATAACCESSOR; // uncomment for ARFF data format
	typedef FST::Classifier_LIBSVM<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> CLASSIFIERSVM;
	typedef FST::Criterion_Wrapper<RETURNTYPE,SUBSET,CLASSIFIERSVM,DATAACCESSOR> WRAPPERSVM;
	typedef FST::Sequential_Step_Straight<RETURNTYPE,DIMTYPE,SUBSET,WRAPPERSVM> EVALUATOR;

		std::cout << "Starting Example 62: (Missing data substitution) Combined feature subset contents, size and SVM parameters optimization..." << std::endl;
	// randomly sample 50% of data for training and randomly sample (disjunct) 40% for independent testing of final classification performance 	
		PSPLITTER dsp_outer(new SPLITTERRR(1, 50, 40)); // (there will be one outer randomized split only)
	// in the course of search use the first half of data by 3-fold cross-validation in wrapper FS criterion evaluation
		PSPLITTER dsp_inner(new SPLITTERCV(3));
	// do not scale data
		const DATATYPE missing_value_code=5;
		boost::shared_ptr<FST::Data_Scaler<DATATYPE> > dsc(new FST::Data_Scaler_void<DATATYPE>(missing_value_code,1/*to choose correct constructor*/));
	// set-up data access
		boost::shared_ptr<std::vector<PSPLITTER> > splitters(new std::vector<PSPLITTER>); 
		splitters->push_back(dsp_outer); splitters->push_back(dsp_inner);
		boost::shared_ptr<DATAACCESSOR> da(new DATAACCESSOR("data/sonar_60_missing_data.trn",splitters,dsc));
		da->initialize();
	// initiate access to split data parts
		da->setSplittingDepth(0); if(!da->getFirstSplit()) throw FST::fst_error("50/40 random data split failed.");
		da->setSplittingDepth(1); if(!da->getFirstSplit()) throw FST::fst_error("3-fold cross-validation failure.");
	// initiate the storage for subset to-be-selected + another one as temporary storage
		boost::shared_ptr<SUBSET> sub(new SUBSET(da->getNoOfFeatures()));
		boost::shared_ptr<SUBSET> sub_temp(new SUBSET(da->getNoOfFeatures()));
	// set-up SVM (interface to external library LibSVM)
		boost::shared_ptr<CLASSIFIERSVM> csvm(new CLASSIFIERSVM);
		csvm->set_kernel_type(RBF); // (option: LINEAR, RBF, POLY)
		csvm->initialize(da);
	// wrap the SVM classifier to enable its usage as FS criterion (criterion value will be estimated by 3-fold cross-val.)
		boost::shared_ptr<WRAPPERSVM> wsvm(new WRAPPERSVM);
		wsvm->initialize(csvm,da);
	// set-up the standard sequential search step object (option: hybrid, ensemble)
		boost::shared_ptr<EVALUATOR> eval(new EVALUATOR);
	// set-up Dynamic Oscillating Search procedure
		FST::Search_DOS<RETURNTYPE,DIMTYPE,SUBSET,WRAPPERSVM,EVALUATOR> srch(eval);
		srch.set_delta(3);
	// run the search
		std::cout << "Feature selection setup:" << std::endl << *da << std::endl << srch << std::endl << *wsvm << std::endl << std::endl;
		RETURNTYPE bestcritval_train, critval_train, critval_test;
		sub->select_all();
		csvm->optimize_parameters(da,sub);
		double best_svm_param_C=csvm->get_parameter_C();
		double best_svm_param_gamma=csvm->get_parameter_gamma();
		double best_svm_param_coef0=csvm->get_parameter_coef0();
		bool stop=false;
		sub->deselect_all();
		if(!srch.search(0,bestcritval_train,sub,wsvm, std::cout)) throw FST::fst_error("Search not finished.");
		sub_temp->stateless_copy(*sub);
		while(!stop) 
		{
			csvm->optimize_parameters(da,sub);
			if(!srch.search(0,critval_train,sub_temp,wsvm,std::cout)) throw FST::fst_error("Search not finished.");
			if(critval_train>bestcritval_train)
			{
				bestcritval_train=critval_train;
				sub->stateless_copy(*sub_temp);
				best_svm_param_C=csvm->get_parameter_C();
				best_svm_param_gamma=csvm->get_parameter_gamma();
				best_svm_param_coef0=csvm->get_parameter_coef0();
			} else stop=true;
		}
		std::cout << std::endl << "Search result: " << std::endl << *sub << std::endl << "Criterion value=" << bestcritval_train << std::endl << std::endl;
	// (optionally) validate result by estimating SVM accuracy on selected feature sub-space on independent test data
		da->setSplittingDepth(0);
		csvm->set_parameter_C(best_svm_param_C);
		csvm->set_parameter_gamma(best_svm_param_gamma);
		csvm->set_parameter_coef0(best_svm_param_coef0);
		csvm->train(da,sub);
		csvm->test(critval_test,da);
		std::cout << "Validated SVM accuracy=" << critval_test << std::endl << std::endl;
	}
	catch(FST::fst_error &e) {std::cerr<<"FST ERROR: "<< e.what() << ", code=" << e.code() << std::endl;}
	catch(std::exception &e) {std::cerr<<"non-FST ERROR: "<< e.what() << std::endl;}
	return 0;
}
