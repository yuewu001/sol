/* =========================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
*/ /*! 

   \file    demo54.cpp 
   \brief   Example 54: Feature Selection Stability Evaluation
   \example demo54.cpp 
            Implements \ref example54, see also 
            <a href="http://fst.utia.cz/?fst3_usage#example54">Example 54 source code</a>
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
#include "search_seq_sfs.hpp"
//#include "search_seq_sffs.hpp"
//#include "search_seq_sfrs.hpp"
//#include "search_seq_os.hpp"
//#include "search_seq_dos.hpp"
//#include "result_tracker_dupless.hpp"
//#include "result_tracker_regularizer.hpp"
//#include "result_tracker_feature_stats.hpp"
#include "result_tracker_stabileval.hpp"


/*! \section example54 Example 54: Feature Selection Stability Evaluation

Feature selection stability is defined as robustness of feature preferences with 
respect to different sampling of the same training data. Different stability measures
have been defined to evaluate the feature selection process (Kalousis, 2007) 
(Somol, IEEE, TPAMI, 2010). FST3 provides a selection of stability measures capable 
of evaluating feature selection process that yields
subsets of varying subset size. In this example ten feature selection trials are 
performed, each on differently randomly sampled 80% of the data. In each of the trials
the resulting subset is obtained through SFS procedure, optimizing the 3-Nearest
Neighbour accuracy evaluated by means of 3-fold cross-validation. Various stability
measures are eventually evaluated. Note that each of the measures yields values from [0,1],
where 0 marks the least stable and 1 marks the most stable feature selection process
characteritic.
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
	typedef FST::Data_Splitter_RandomRandom<INTERVALLER,IDXTYPE,BINTYPE> SPLITTERRANDRAND;
	typedef FST::Data_Accessor_Splitting_MemTRN<DATATYPE,IDXTYPE,INTERVALLER> DATAACCESSOR; // uncomment for TRN data format
	//typedef FST::Data_Accessor_Splitting_MemARFF<DATATYPE,IDXTYPE,INTERVALLER> DATAACCESSOR; // uncomment for ARFF data format
	typedef FST::Distance_Euclid<DATATYPE,DIMTYPE,SUBSET> DISTANCE;
	typedef FST::Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE> CLASSIFIERKNN;
	typedef FST::Criterion_Wrapper<RETURNTYPE,SUBSET,CLASSIFIERKNN,DATAACCESSOR> WRAPPERKNN;
	typedef FST::Sequential_Step_Straight<RETURNTYPE,DIMTYPE,SUBSET,WRAPPERKNN> EVALUATOR;
	typedef FST::Result_Tracker_Stability_Evaluator<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET> TRACKER;

		std::cout << "Starting Example 54: Feature Selection Stability Evaluation..." << std::endl;
	// set-up ten trials where in each 80% of data is randomly sampled
		PSPLITTER dsp_outer(new SPLITTERRANDRAND(10/*splits=trials*/,80,20));
	// in the course of feature subset search (in one trial) use 3-fold cross-validation
		PSPLITTER dsp_inner(new SPLITTERCV(3));
	// do not scale data
		boost::shared_ptr<FST::Data_Scaler<DATATYPE> > dsc(new FST::Data_Scaler_void<DATATYPE>());
	// set-up data access
		boost::shared_ptr<std::vector<PSPLITTER> > splitters(new std::vector<PSPLITTER>); 
		splitters->push_back(dsp_outer); splitters->push_back(dsp_inner);
		boost::shared_ptr<DATAACCESSOR> da(new DATAACCESSOR("data/speech_15.trn",splitters,dsc));
		da->initialize();
	// initiate access to split data parts
		da->setSplittingDepth(0); if(!da->getFirstSplit()) throw FST::fst_error("RandRand data split failed.");
		da->setSplittingDepth(1); if(!da->getFirstSplit()) throw FST::fst_error("3-fold cross-validation failure.");
	// initiate the storage for subset to-be-selected
		boost::shared_ptr<SUBSET> sub(new SUBSET(da->getNoOfFeatures()));  sub->deselect_all();
	// set-up 3-Nearest Neighbor classifier based on Euclidean distances
		boost::shared_ptr<CLASSIFIERKNN> cknn(new CLASSIFIERKNN); cknn->set_k(3);
	// wrap the 3-NN classifier to enable its usage as FS criterion (criterion value will be estimated by 3-fold cross-val.)
		boost::shared_ptr<WRAPPERKNN> wknn(new WRAPPERKNN);
		wknn->initialize(cknn,da);
	// set-up the standard sequential search step object (option: hybrid, ensemble, threaded, etc.)
		boost::shared_ptr<EVALUATOR> eval(new EVALUATOR);
	// set-up Sequential Forward Floating Selection search procedure
		FST::Search_SFS<RETURNTYPE,DIMTYPE,SUBSET,WRAPPERKNN,EVALUATOR> srch(eval);
		srch.set_search_direction(FST::FORWARD);
	// set-up result tracker to collect results of each trial
		boost::shared_ptr<TRACKER> tracker(new TRACKER);
	// run the trials
		std::cout << "Feature selection setup:" << std::endl << *da << std::endl << srch << std::endl << *wknn << std::endl << *tracker << std::endl << std::endl;
		RETURNTYPE critval_train;
		da->setSplittingDepth(0);
		unsigned int trial=0;
		bool run=da->getFirstSplit(); if(!run) throw FST::fst_error("RandRand data split failed.");
		while(run)
		{
			trial++; std::cout << std::endl<<"TRIAL "<<trial<< " ---------------------------------------------------------------------"<<std::endl;
			da->setSplittingDepth(1);
			if(!srch.search(0,critval_train,sub,wknn,std::cout)) throw FST::fst_error("Search not finished.");
			tracker->add(critval_train,sub);
			std::cout << std::endl << "(TRIAL "<<trial<<") Search result: " << std::endl << *sub << "Criterion value=" << critval_train << std::endl;
			da->setSplittingDepth(0);
			run=da->getNextSplit();
		}
	// evaluate stability using results collected by tracker
		std::cout<<std::endl;
		std::cout << "---------------------------------------------------------------------" << std::endl;
		std::cout << "Resulting criterion values' mean: " << tracker->value_mean() << ", standard deviation: " << tracker->value_stddev() << std::endl;
		std::cout << "Resulting subset sizes' mean: " << tracker->size_mean() << ", standard deviation: " << tracker->size_stddev() << std::endl;
		std::cout << std::endl;
		std::cout << "stability measure CWrel("<<da->getNoOfFeatures()<<")=" << tracker->stability_CWrel(da->getNoOfFeatures()) << std::endl;
		std::cout << "stability measure ATI()=" << tracker->stability_ATI() << std::endl;
		std::cout << "stability measure CW()=" << tracker->stability_CW() << std::endl;
		std::cout << "stability measure ANHI("<<da->getNoOfFeatures()<<")=" << tracker->stability_ANHI(da->getNoOfFeatures()) << std::endl;
		std::cout << "stability measure SH()=" << tracker->stability_SH() << std::endl;
		std::cout << "stability measure PH("<<da->getNoOfFeatures()<<")=" << tracker->stability_PH(da->getNoOfFeatures()) << std::endl;
		std::cout << "stability measure C()=" << tracker->stability_C() << std::endl;
	}
	catch(FST::fst_error &e) {std::cerr<<"FST ERROR: "<< e.what() << ", code=" << e.code() << std::endl;}
	catch(std::exception &e) {std::cerr<<"non-FST ERROR: "<< e.what() << std::endl;}
	return 0;
}
