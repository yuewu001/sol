/* =========================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
*/ /*! 

   \file    demo55.cpp 
   \brief   Example 55: Evaluating Similarity of Two Feature Selection Processes
   \example demo55.cpp 
            Implements \ref example55, see also 
            <a href="http://fst.utia.cz/?fst3_usage#example55">Example 55 source code</a>
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
#include "distance_L1.hpp"
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
#include "search_seq_sffs.hpp"
//#include "search_seq_sfrs.hpp"
//#include "search_seq_os.hpp"
#include "search_seq_dos.hpp"
//#include "result_tracker_dupless.hpp"
//#include "result_tracker_regularizer.hpp"
//#include "result_tracker_feature_stats.hpp"
#include "result_tracker_stabileval.hpp"


/*! \section example55 Example 55: Evaluating Similarity of Two Feature Selection Processes.

To study the difference in feature preferences among principally different feature selection methods
or among differently parametrized instances of the same method FST3 provides measures
capable of evaluating the level of similarity between two sets of trials (Somol Novovicova, IEEE, TPAMI, 2010). 
In analogy to stability evaluation (see \ref example54) for each of the two feature selection scenarios 
a series of trials is conducted on various samplings of the same data. In this example ten feature selection 
trials are performed per scenario, each on randomly sampled 95% of the data. In the first scenario 
in each trial the resulting subset is obtained using DOS procedure, optimizing the 3-Nearest
Neighbour accuracy estimated by means of 3-fold cross-validation. In the second scenario
in each trial the resulting subset is obtained using SFFS procedure, maximizing
the Bhattacharyya distance based on normal model. A selection of standard stability measures
is evaluated separately for each of the two scenarios. Eventually the similarity of the two
scenarios is evaluated using analogously founded similarity measures. All measures yield values from
[0,1], where values close to 0 denote low stability/similarity and values close to 1 denote 
high stability/similarity. Note that in this experiment the inter-measures (IATI, ICW, IANHI) 
yield markedly lower values than the corresponding stability measures (ATI, CW, ANHI). 
This illustrates well that considerably different results can be expected from differently founded
feature selection methods.
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
	typedef FST::Distance_L1<DATATYPE,DIMTYPE,SUBSET> DISTANCEL1;
	typedef FST::Classifier_kNN<RETURNTYPE,DATATYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR,DISTANCEL1> CLASSIFIERKNN;
	typedef FST::Criterion_Wrapper<RETURNTYPE,SUBSET,CLASSIFIERKNN,DATAACCESSOR> WRAPPER;
	typedef FST::Sequential_Step_Straight<RETURNTYPE,DIMTYPE,SUBSET,WRAPPER> EVALUATOR1;
	typedef FST::Criterion_Normal_Bhattacharyya<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> BHATTCRIT;
	typedef FST::Sequential_Step_Straight<RETURNTYPE,DIMTYPE,SUBSET,BHATTCRIT> EVALUATOR2;
	typedef FST::Result_Tracker_Stability_Evaluator<RETURNTYPE,IDXTYPE,DIMTYPE,SUBSET> TRACKER;

		std::cout << "Starting Example 55: Evaluating Similarity of Two Feature Selection Processes..." << std::endl;
	// set-up ten trials where in each 95% of data is randomly sampled
		PSPLITTER dsp_outer(new SPLITTERRANDRAND(10/*splits=trials*/,95,5));
	// in the course of wrapper based feature subset search (in one trial) use 3-fold cross-validation
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
	// set-up result trackers to collect results of each trial in both scenarios
		boost::shared_ptr<TRACKER> tracker1(new TRACKER);
		boost::shared_ptr<TRACKER> tracker2(new TRACKER);
	// FEATURE SELECTION SCENARIO A (wrapper)
	// set-up 3-Nearest Neighbor classifier based on L1 distances
		boost::shared_ptr<CLASSIFIERKNN> cknn1(new CLASSIFIERKNN);
		cknn1->set_k(3);
	// wrap the 3-NN classifier to enable its usage as FS criterion (criterion value will be estimated by 3-fold cross-val.)
		boost::shared_ptr<WRAPPER> wknn1(new WRAPPER);
		wknn1->initialize(cknn1,da);
	// set-up the standard sequential search step object (option: hybrid, ensemble, threaded)
		boost::shared_ptr<EVALUATOR1> eval1(new EVALUATOR1);
	// set-up Sequential Forward Floating Selection search procedure
		FST::Search_DOS<RETURNTYPE,DIMTYPE,SUBSET,WRAPPER,EVALUATOR1> srch1(eval1);
		srch1.set_delta(10);
		sub->deselect_all();
		// Technical remark: should threaded evaluator be used in this case, it would be necessary to move both the evaluator and search procedure set-up 
		// inside the trial loop. The reason is technical: threaded evaluator caches criterion clones, including data accessor state. 
		// Therefore no outside changes in splitting level nor current split change can be reflected in criterion evaluation. Renewed 
		// evaluator set-up resets the cache and thus ensures correct threaded criterion evaluation behavior after split change.
	// run the trials
		std::cout << "Feature selection setup:" << std::endl << *da << std::endl << *wknn1 << std::endl << *tracker1 << std::endl << std::endl;
		RETURNTYPE critval_train;
		da->setSplittingDepth(0);
		unsigned int trial=0;
		bool run=da->getFirstSplit(); if(!run) throw FST::fst_error("RandRand data split failed.");
		while(run)
		{
			trial++; std::cout << std::endl<<"TRIAL A"<<trial<< " ---------------------------------------------------------------------"<<std::endl;
			da->setSplittingDepth(1);
			if(!srch1.search(0,critval_train,sub,wknn1,std::cout)) throw FST::fst_error("Search not finished.");
			tracker1->add(critval_train,sub);
			std::cout << std::endl << "(TRIAL A"<<trial<<") Search result: " << std::endl << *sub << "Criterion value=" << critval_train << std::endl;
			da->setSplittingDepth(0);
			run=da->getNextSplit();
		}
	// FEATURE SELECTION SCENARIO B (filter)
	// set-up normal Bhattacharyya distance criterion 
		boost::shared_ptr<BHATTCRIT> cb(new BHATTCRIT);
	// set-up the standard sequential search step object (option: hybrid, ensemble, threaded)
		boost::shared_ptr<EVALUATOR2> eval2(new EVALUATOR2);
	// set-up Sequential Forward Floating Selection search procedure
		FST::Search_SFFS<RETURNTYPE,DIMTYPE,SUBSET,BHATTCRIT,EVALUATOR2> srch2(eval2);
		srch2.set_search_direction(FST::FORWARD);
	// target subset size must be set because Bhattacharyya is monotonous with respect to subset size (i.e., evaluates full set as the best)
		const DIMTYPE target_size=7;
	// run the trials
		std::cout << "Feature selection setup:" << std::endl << *da << std::endl << srch2 << std::endl << *cb << std::endl << *tracker2 << std::endl << std::endl;
		trial=0;
		da->setSplittingDepth(0);
		run=da->getFirstSplit(); if(!run) throw FST::fst_error("RandRand data split failed.");
		while(run)
		{
			trial++; std::cout << std::endl<<"TRIAL B"<<trial<< " ---------------------------------------------------------------------"<<std::endl;
			cb->initialize(da); // (note that cb initialization = normal model parameter estimation on training data, therefore it must be repeated for each split)
			da->setSplittingDepth(1);
			if(!srch2.search(target_size,critval_train,sub,cb,std::cout)) throw FST::fst_error("Search not finished.");
			tracker2->add(critval_train,sub);
			std::cout << std::endl << "(TRIAL B"<<trial<<") Search result: " << std::endl << *sub << "Criterion value=" << critval_train << std::endl;
			da->setSplittingDepth(0);
			run=da->getNextSplit();
		}
	// evaluate stability of each scenario and similarity of the two scenarios using results collected by trackers
		std::cout<<std::endl;
		std::cout << "---------------------------------------------------------------------" << std::endl;
		std::cout << "Scenario A resulting criterion values' mean: " << tracker1->value_mean() << ", std. dev.: " << tracker1->value_stddev() << std::endl;
		std::cout << "Scenario A subset sizes' mean: " << tracker1->size_mean() << ", std. dev.: " << tracker1->size_stddev() << std::endl;
		std::cout << std::endl;
		std::cout << "Scenario A stability_ATI()=" << tracker1->stability_ATI() << std::endl;
		std::cout << "Scenario A stability_CW()=" << tracker1->stability_CW() << std::endl;
		std::cout << "Scenario A stability_ANHI("<<da->getNoOfFeatures()<<")=" << tracker1->stability_ANHI(da->getNoOfFeatures()) << std::endl;
		std::cout<<std::endl;
		std::cout << "Scenario B resulting criterion values' mean: " << tracker2->value_mean() << ", std. dev.: " << tracker2->value_stddev() << std::endl;
		std::cout << "Scenario B subset sizes' mean: " << tracker2->size_mean() << ", std. dev.: " << tracker2->size_stddev() << std::endl;
		std::cout << std::endl;
		std::cout << "Scenario B stability_ATI()=" << tracker2->stability_ATI() << std::endl;
		std::cout << "Scenario B stability_CW()=" << tracker2->stability_CW() << std::endl;
		std::cout << "Scenario B stability_ANHI("<<da->getNoOfFeatures()<<")=" << tracker2->stability_ANHI(da->getNoOfFeatures()) << std::endl;
		std::cout<<std::endl;
		std::cout << "Evaluating similarity between scenario A and scenario B:"<< std::endl;
		std::cout << "similarity measure IATI()=" << tracker1->similarity_IATI(*tracker2) << std::endl;
		std::cout << "similarity measure ICW()=" << tracker1->similarity_ICW(*tracker2) << std::endl;
		std::cout << "similarity measure IANHI("<<da->getNoOfFeatures()<<")=" << tracker1->similarity_IANHI(da->getNoOfFeatures(), *tracker2) << std::endl;
	}
	catch(FST::fst_error &e) {std::cerr<<"FST ERROR: "<< e.what() << ", code=" << e.code() << std::endl;}
	catch(std::exception &e) {std::cerr<<"non-FST ERROR: "<< e.what() << std::endl;}
	return 0;
}
