/* =========================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
*/ /*! 

   \file    demo22.cpp 
   \brief   Example 22: Randomized feature selection with Oscillating Search
   \example demo22.cpp 
            Implements \ref example22, see also 
            <a href="http://fst.utia.cz/?fst3_usage#example22">Example 22 source code</a>
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
//#include "data_splitter_cv.hpp"
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

#include "criterion_normal_bhattacharyya.hpp"
//#include "criterion_normal_gmahalanobis.hpp"
//#include "criterion_normal_divergence.hpp"
//#include "criterion_multinom_bhattacharyya.hpp"
#include "criterion_wrapper.hpp"
//#include "criterion_wrapper_bias_estimate.hpp"
//#include "criterion_subsetsize.hpp"
//#include "criterion_sumofweights.hpp"
//#include "criterion_negative.hpp"

#include "distance_euclid.hpp"
//#include "distance_L1.hpp"
//#include "distance_Lp.hpp"
#include "classifier_knn.hpp"
//#include "classifier_normal_bayes.hpp"
//#include "classifier_multinom_naivebayes.hpp"
//#include "classifier_svm.hpp"

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
#include "search_seq_os.hpp"
//#include "search_seq_dos.hpp"
//#include "result_tracker_dupless.hpp"
//#include "result_tracker_regularizer.hpp"
//#include "result_tracker_feature_stats.hpp"
//#include "result_tracker_stabileval.hpp"


/*! \section example22 Example 22: Randomized feature selection with Oscillating Search. 

Selects features using randomized OS algorithm and normal Bhattacharyya distance as feature selection criterion. Target 
subset size must be set by user because a) Bhattacharyya is monotonous (global optimization would yield full set 
of features) and b) OS requires initial subset of target cardinality. OS is called here repeatedly from 
various random initial points as long as there comes no improvement during consecutive 5 runs. This mechanism 
considerably improves chances to avoid local maxima. Bhattacharyya is evaluated on randomly chosen 75% of data samples. 
The resulting feature subset is eventually validated by means of 3NN classification accuracy estimation on
on the remaining 25% of data samples.
*/
int main()
{
	try{
	typedef double RETURNTYPE; 	typedef double DATATYPE;  typedef double REALTYPE;
	typedef unsigned int IDXTYPE;  typedef unsigned int DIMTYPE;  typedef short BINTYPE;
	typedef FST::Subset<BINTYPE, DIMTYPE> SUBSET;
	typedef FST::Data_Intervaller<std::vector<FST::Data_Interval<IDXTYPE> >,IDXTYPE> INTERVALLER;
	typedef boost::shared_ptr<FST::Data_Splitter<INTERVALLER,IDXTYPE> > PSPLITTER;
	typedef FST::Data_Splitter_RandomRandom<INTERVALLER,IDXTYPE,BINTYPE> SPLITTERRANDRAND;
	typedef FST::Data_Accessor_Splitting_MemTRN<DATATYPE,IDXTYPE,INTERVALLER> DATAACCESSOR; // uncomment for TRN data format
	//typedef FST::Data_Accessor_Splitting_MemARFF<DATATYPE,IDXTYPE,INTERVALLER> DATAACCESSOR; // uncomment for ARFF data format
	typedef FST::Criterion_Normal_Bhattacharyya<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> BHATTCRIT;
	typedef FST::Distance_Euclid<DATATYPE,DIMTYPE,SUBSET> DISTANCE;
	typedef FST::Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE> CLASSIFIERKNN;
	typedef FST::Criterion_Wrapper<RETURNTYPE,SUBSET,CLASSIFIERKNN,DATAACCESSOR> WRAPPERKNN;
	typedef FST::Sequential_Step_Straight<RETURNTYPE,DIMTYPE,SUBSET,BHATTCRIT> EVALUATOR;

		std::cout << "Starting Example 22: Randomized feature selection with Oscillating Search..." << std::endl;
	// use randomly chosen 75% of data for training and keep the other 25% for independent testing of final classification performance
		PSPLITTER dsp_outer(new SPLITTERRANDRAND(1,75,25)); // (there will be one outer randomized split only)
	// do not scale data
		boost::shared_ptr<FST::Data_Scaler<DATATYPE> > dsc(new FST::Data_Scaler_void<DATATYPE>());
	// set-up data access
		boost::shared_ptr<std::vector<PSPLITTER> > splitters(new std::vector<PSPLITTER>); 
		splitters->push_back(dsp_outer); 
		boost::shared_ptr<DATAACCESSOR> da(new DATAACCESSOR("data/sonar_60.trn",splitters,dsc));
		da->initialize();
	// initiate access to split data parts
		da->setSplittingDepth(0); if(!da->getFirstSplit()) throw FST::fst_error("75/25 data split failed.");
	// initiate the storage for subset to-be-selected (+ one more for storing temporaries)
		boost::shared_ptr<SUBSET> sub(new SUBSET(da->getNoOfFeatures()));
		boost::shared_ptr<SUBSET> submax(new SUBSET(da->getNoOfFeatures()));
	// set-up normal Bhattacharyya criterion 
		boost::shared_ptr<BHATTCRIT> cb(new BHATTCRIT);
		cb->initialize(da); // initialization = normal model parameter estimation on training data on the current split
	// set-up the standard sequential search step object (option: hybrid, ensemble, etc.)
		boost::shared_ptr<EVALUATOR> eval(new EVALUATOR);
	// set-up Oscillating Search procedure with the extent of search specified by delta (admissible values 1..da->getNoOfFeatures())
		FST::Search_OS<RETURNTYPE,DIMTYPE,SUBSET,BHATTCRIT,EVALUATOR> srch(eval);
		srch.set_delta(10);
	// target subset size must be set because Bhattacharyya is monotonous with respect to subset size (i.e., evaluates full set as the best)
		DIMTYPE target_subsize=30;
	// repeat random-initialized OS searches until there is no improvement in 5 consecutive runs
		std::cout << "Feature selection setup:" << std::endl << *da << std::endl << srch << std::endl << *cb << std::endl << std::endl;
		RETURNTYPE critval_train, critval_trainmax;
		int non_improving_runs=-1; // -1 indicates first run
		const int max_non_improving_runs=5;
		do {
			sub->make_random_subset(target_subsize);
			if(!srch.search(target_subsize,critval_train,sub,cb,std::cout)) throw FST::fst_error("Search not finished.");
			if(non_improving_runs==-1 || critval_train>critval_trainmax) {
				critval_trainmax=critval_train;
				submax->stateless_copy(*sub);
				non_improving_runs=0;
			} else non_improving_runs++;
		} while(non_improving_runs<max_non_improving_runs-1);
		std::cout << std::endl << "Final search result: " << std::endl << *submax << std::endl << "Criterion value=" << critval_trainmax << std::endl << std::endl;
	// (optionally) validate result by estimating kNN accuracy on selected feature sub-space on independent test data
		RETURNTYPE critval_test;
		boost::shared_ptr<CLASSIFIERKNN> cknn(new CLASSIFIERKNN); cknn->set_k(3);
		cknn->train(da,submax);
		cknn->test(critval_test,da);
		std::cout << "Validated "<<cknn->get_k()<<"-NN accuracy=" << critval_test << std::endl << std::endl;
	}
	catch(FST::fst_error &e) {std::cerr<<"FST ERROR: "<< e.what() << ", code=" << e.code() << std::endl;}
	catch(std::exception &e) {std::cerr<<"non-FST ERROR: "<< e.what() << std::endl;}
	return 0;
}
