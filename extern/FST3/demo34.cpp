/* =========================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
*/ /*! 

   \file    demo34.cpp 
   \brief   Example 34: Dependency-Aware Feature Ranking (DAF0)
   \example demo34.cpp 
            Implements \ref example34, see also 
            <a href="http://fst.utia.cz/?fst3_usage#example34">Example 34 source code</a>
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see 
            <a href="http://fst.utia.cz/?contacts">Contacts</a>
   \date    March 2011
   \version 3.1.0.beta
   \note    FST3 was developed using gcc 4.3 and requires
   \note    \li Boost library (http://www.boost.org/, tested with versions 1.33.1 and 1.44),
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
    of FST3, or if in doubt, to FST3 maintainer (see http://fst.utia.cz).
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

#include "distance_euclid.hpp"
//#include "distance_L1.hpp"
//#include "distance_Lp.hpp"
#include "classifier_knn.hpp"
//#include "classifier_normal_bayes.hpp"
//#include "classifier_multinom_naivebayes.hpp"
//#include "classifier_svm.hpp"

//#include "search_bif.hpp"
//#include "search_bif_threaded.hpp"
#include "search_monte_carlo.hpp"
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
//#include "seq_step_hybrid.hpp"
//#include "seq_step_ensemble.hpp"
//#include "search_seq_sfs.hpp"
//#include "search_seq_sffs.hpp"
//#include "search_seq_sfrs.hpp"
//#include "search_seq_os.hpp"
//#include "search_seq_dos.hpp"
//#include "result_tracker_dupless.hpp"
//#include "result_tracker_regularizer.hpp"
#include "result_tracker_feature_stats.hpp"
//#include "result_tracker_stabileval.hpp"


/*! \section example34 Example 34: Dependency-Aware Feature Ranking (DAF0).

Dependency-Aware Feature Ranking (DAF) is a novel approach to feature
selection especially suitable for very-high-dimensional problems
and over-fitting-prone feature selection scenarios.
DAF evaluates a chosen criterion on a series of probe subsets
to eventually rank features according to their estimated contextual quality.
DAF has been shown capable of overperforming BIF quite significantly in many
cases in terms of the quality of selected feature subsets, yet its stability 
and resistance against over-fitting remains on par with BIF.
For details see UTIA Technical Report No. 2295 from February 2011.
We demonstrate two slightly different forms of DAF (DAF0 and DAF1)
on examples \ref example34 and \ref example35t. Example34
illustrates the approach with k-NN accuracy \e wrapper criterion
on 64-dimensional image data. Example 35t illustrates DAF with SVM \e wrapper applied 
to very-high-dimensional text categorization problem.

\note DAF (as BIF) provides feature ranking but does not determine final subset size.

\note This approach makes it possible to apply even the complex
\e wrapper feature selection criteria in problems of very-high-dimensionality.

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
	typedef FST::Data_Splitter_5050<INTERVALLER,IDXTYPE> SPLITTER5050;
	typedef FST::Data_Accessor_Splitting_MemTRN<DATATYPE,IDXTYPE,INTERVALLER> DATAACCESSOR; // uncomment for TRN data format
	//typedef FST::Data_Accessor_Splitting_MemARFF<DATATYPE,IDXTYPE,INTERVALLER> DATAACCESSOR; // uncomment for ARFF data format
	typedef FST::Distance_Euclid<DATATYPE,DIMTYPE,SUBSET> DISTANCE;
	typedef FST::Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCE> CLASSIFIERKNN;
	typedef FST::Criterion_Wrapper<RETURNTYPE,SUBSET,CLASSIFIERKNN,DATAACCESSOR> WRAPPERKNN;
	typedef FST::Result_Tracker_Feature_Stats<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET> TRACKERSTATS;

		std::cout << "Starting Example 34: Dependency-Aware Feature Ranking (DAF0)..." << std::endl;
	// keep second half of data for independent testing of final classification performance
		PSPLITTER dsp_outer(new SPLITTER5050());
	// in the course of search use the first half of data by 3-fold cross-validation in wrapper FS criterion evaluation
		PSPLITTER dsp_inner(new SPLITTERCV(3));
	// do not scale data
		boost::shared_ptr<FST::Data_Scaler<DATATYPE> > dsc(new FST::Data_Scaler_void<DATATYPE>());
	// set-up data access
		boost::shared_ptr<std::vector<PSPLITTER> > splitters(new std::vector<PSPLITTER>); 
		splitters->push_back(dsp_outer); splitters->push_back(dsp_inner);
		boost::shared_ptr<DATAACCESSOR> da(new DATAACCESSOR("data/optdigits_64.trn",splitters,dsc));
		da->initialize();
	// initiate access to split data parts
		da->setSplittingDepth(0); if(!da->getFirstSplit()) throw FST::fst_error("50/50 data split failed.");
		da->setSplittingDepth(1); if(!da->getFirstSplit()) throw FST::fst_error("3-fold cross-validation failure.");
	// initiate the storage for subset to-be-selected
		boost::shared_ptr<SUBSET> sub(new SUBSET(da->getNoOfFeatures()));  sub->deselect_all();
	// set-up 3-Nearest Neighbor classifier based on Euclidean distances
		boost::shared_ptr<CLASSIFIERKNN> cknn(new CLASSIFIERKNN); cknn->set_k(3);
	// wrap the 3-NN classifier to enable its usage as FS criterion (criterion value will be estimated by 3-fold cross-val.)
		boost::shared_ptr<WRAPPERKNN> wknn(new WRAPPERKNN);
		wknn->initialize(cknn,da);
	// Dependency-Aware Feature ranking computation settings
		const unsigned long max_search_time=600; // in seconds (the more search time can be afforded the better; for very-high-dimensional problems set better to hours or more)
		const DIMTYPE min_probe_cardinality=5; // lower limit on random probe subset cardinality (the default value is generally applicable)
		const DIMTYPE max_probe_cardinality=100; // upper limit on random probe subset cardinality (the default value is generally applicable)
	// set-up the probe subset generation procedure
		FST::Search_Monte_Carlo<RETURNTYPE,DIMTYPE,SUBSET,WRAPPERKNN> srch;
		srch.set_cardinality_randomization(min_probe_cardinality,max_probe_cardinality);
		srch.set_stopping_condition(0/*max trials*/,max_search_time/*seconds*/); // one or both values must have positive value
	// set-up tracker to gather data for eventual DAF rank computation
		boost::shared_ptr<TRACKERSTATS> trackerstats(new TRACKERSTATS);
		srch.enable_result_tracking(trackerstats);
	// run the search
		std::cout << "Feature selection setup:" << std::endl << *da << std::endl << srch << std::endl << *wknn << std::endl << std::endl;
		RETURNTYPE critval_train, critval_test;
		srch.set_output_detail(FST::NORMAL); // set FST::SILENT to disable all text output in the course of search (FST::NORMAL is default)
		if(!srch.search(0,critval_train,sub,wknn,std::cout)) throw FST::fst_error("Search not finished.");
	// compute DAF ranking
		trackerstats->compute_stats();
	// (optionally) print DAF computation statistics
		trackerstats->print_stats(std::cout);
	// select user-specified number of features according to highest DAF feature rank values
	// + validate result by estimating classifier accuracy on selected feature sub-space on independent test data
		da->setSplittingDepth(0);
		const DIMTYPE d=60; 
		unsigned int DAF=0; // DAF0 is the simplest and generally best performing option; DAF1 as a normalized version of DAF0 may occasionally yield better results
		RETURNTYPE critval;
		DIMTYPE i=0, feature;
		sub->deselect_all();
		bool found=trackerstats->getFirstDAF(critval,feature,DAF);
		while(i++<d && found) {
			sub->select(feature);
			std::cout << "Added feature "<<feature<<", DAF"<<DAF<<"=" << critval << std::endl;
			if(i%5==0) { // (optionally) validate result by estimating classifier accuracy on selected feature sub-space on independent test data
				cknn->train(da,sub);
				cknn->test(critval_test,da);
				std::cout << *sub << std::endl << "Validated "<<cknn->get_k()<<"-NN accuracy=" << critval_test << std::endl << std::endl;
			}
			found=trackerstats->getNextDAF(critval,feature,DAF);
		}
	}
	catch(FST::fst_error &e) {std::cerr<<"FST ERROR: "<< e.what() << ", code=" << e.code() << std::endl;}
	catch(std::exception &e) {std::cerr<<"non-FST ERROR: "<< e.what() << std::endl;}
	return 0;
}
