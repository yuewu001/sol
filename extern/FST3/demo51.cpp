/* =========================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
*/ /*! 

   \file    demo51.cpp 
   \brief   Example 51: (DOS) Result regularization using secondary criterion
   \example demo51.cpp 
            Implements \ref example51, see also 
            <a href="http://fst.utia.cz/?fst3_usage#example51">Example 51 source code</a>
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
#include "criterion_subsetsize.hpp"
//#include "criterion_sumofweights.hpp"
#include "criterion_negative.hpp"

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
//#include "search_seq_os.hpp"
#include "search_seq_dos.hpp"
//#include "result_tracker_dupless.hpp"
#include "result_tracker_regularizer.hpp"
//#include "result_tracker_feature_stats.hpp"
//#include "result_tracker_stabileval.hpp"


/*! \section example51 Example 51: (DOS) Result regularization using secondary criterion.

It is known that feature selection may over-fit. As in the case of over-trained classifiers,
over-selected feature subsets may generalize poorly. This unwanted effect can lead to
serious degradation of generalization ability, i.e., model or decision-rule behavior 
on previously unknown data. It has been suggested (Raudys: Feature Over-Selection, LNCS 4109, 2006, 
or Somol et al., ICPR 2010) that preferring a subset with slightly-worse-than-maximal criterion
value can actually improve generalization. FST3 makes this possible through result tracking
and subsequent selection of alternative solution by means of secondary criterion maximization.
In this example we show a 3-Nearest Neighbor Wrapper based feature selection process, where
the final result is eventually chosen among a group of solutions close enough to the achieved
maximum, so as to optimize the secondary criterion. The group of solutions to select from is defined 
by means of a user-selected margin value (permitted primary criterion value difference from the known
maximum). In this case we show that even the simplest secondary criterion (mere preference of 
smaller subsets) can improve classifcation accuracy on previously unknown data.
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
	typedef FST::Criterion_Subset_Size<RETURNTYPE,SUBSET> CRITSUBSIZE;
	typedef FST::Criterion_Negative<CRITSUBSIZE,RETURNTYPE,SUBSET> NEGATIVECRIT;
	typedef FST::Sequential_Step_Straight<RETURNTYPE,DIMTYPE,SUBSET,WRAPPERKNN> EVALUATOR;
	typedef FST::Result_Tracker_Regularizer<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET,NEGATIVECRIT> TRACKER;

		std::cout << "Starting Example 51: (DOS) Result regularization using secondary criterion..." << std::endl;
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
		boost::shared_ptr<SUBSET> sub(new SUBSET(da->getNoOfFeatures()));  sub->deselect_all();
	// set-up 3-Nearest Neighbor classifier based on Euclidean distances
		boost::shared_ptr<CLASSIFIERKNN> cknn(new CLASSIFIERKNN); cknn->set_k(3);
	// wrap the 3-NN classifier to enable its usage as FS criterion (criterion value will be estimated by 3-fold cross-val.)
		boost::shared_ptr<WRAPPERKNN> wknn(new WRAPPERKNN);
		wknn->initialize(cknn,da);
	// set-up the standard sequential search step object (option: hybrid, ensemble)
		boost::shared_ptr<EVALUATOR> eval(new EVALUATOR);
	// set-up Dynamic Oscillating Search procedure
		FST::Search_DOS<RETURNTYPE,DIMTYPE,SUBSET,WRAPPERKNN,EVALUATOR> srch(eval);
		srch.set_delta(3);
	// set-up the regularizing result tracker
		boost::shared_ptr<TRACKER> tracker(new TRACKER);
	// register the result tracker with the used search step object
		eval->enable_result_tracking(tracker);
	// run the search
		std::cout << "Feature selection setup:" << std::endl << *da << std::endl << srch << std::endl << *wknn << std::endl << *tracker << std::endl << std::endl;
		RETURNTYPE critval_train, critval_test;
		srch.set_output_detail(FST::NORMAL); // set FST::SILENT to disable all text output in the course of search (FST::NORMAL is default)
		if(!srch.search(0,critval_train,sub,wknn,std::cout)) throw FST::fst_error("Search not finished.");
	// (optionally) validate result by estimating kNN accuracy on selected feature sub-space on independent test data
		da->setSplittingDepth(0);
		cknn->train(da,sub);
		cknn->test(critval_test,da);
		std::cout << "Validated "<<cknn->get_k()<<"-NN accuracy=" << critval_test << std::endl << std::endl;
	// set-up the secondary criterion (regularization criterion); in this case to minimize subset size
		boost::shared_ptr<CRITSUBSIZE> critsubsiz(new CRITSUBSIZE); //Criterion_Subset_Size does not need to be initialized
		boost::shared_ptr<NEGATIVECRIT> regulcrit(new NEGATIVECRIT(critsubsiz)); //Criterion_Negative does not need to be initialized
	// select final solution among those recorded by tracker (show more alternatives for various margins)
		tracker->set_output_detail(FST::NORMAL); // set FST::SILENT to disable all text output in the course of search (FST::NORMAL is default)
		for(unsigned int i=1; i<10; i++) 
		{
			RETURNTYPE margin=(double)i*0.001;
			da->setSplittingDepth(1); // necessary with criteria than need access to training data
			if(!tracker->optimize_within_margin(margin,critval_train,critval_test,sub,regulcrit)) throw FST::fst_error("tracker->optimize_within_margin() failed.");
			std::cout << std::endl << "Regularized (margin="<<margin<<") result: " << std::endl << *sub << "Criterion value=" << critval_train << std::endl;
		// (optionally) validate result by estimating kNN accuracy on selected feature sub-space on independent test data
			da->setSplittingDepth(0);
			cknn->train(da,sub);
			cknn->test(critval_test,da);
			std::cout << "Validated "<<cknn->get_k()<<"-NN accuracy=" << critval_test << std::endl << std::endl;
		}
	}
	catch(FST::fst_error &e) {std::cerr<<"FST ERROR: "<< e.what() << ", code=" << e.code() << std::endl;}
	catch(std::exception &e) {std::cerr<<"non-FST ERROR: "<< e.what() << std::endl;}
	return 0;
}
