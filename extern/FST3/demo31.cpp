/* =========================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
   */ /*!

   \file    demo31.cpp
   \brief   Example 31: Individual ranking (BIF) in very high-dimensional feature selection
   \example demo31.cpp
   Implements \ref example31, see also
   <a href="http://fst.utia.cz/?fst3_usage#example31">Example 31 source code</a>
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
#include "data_scaler_to01.hpp"
#include "data_scaler_white.hpp"
#include "data_accessor_splitting_memTRN.hpp"
#include "data_accessor_splitting_memARFF.hpp"

//#include "criterion_normal_bhattacharyya.hpp"
//#include "criterion_normal_gmahalanobis.hpp"
//#include "criterion_normal_divergence.hpp"
#include "criterion_multinom_bhattacharyya.hpp"
#include "criterion_wrapper.hpp"
//#include "criterion_wrapper_bias_estimate.hpp"
//#include "criterion_subsetsize.hpp"
//#include "criterion_sumofweights.hpp"
//#include "criterion_negative.hpp"

//#include "distance_euclid.hpp"
//#include "distance_L1.hpp"
//#include "distance_Lp.hpp"
#include "classifier_knn.hpp"
//#include "classifier_normal_bayes.hpp"
#include "classifier_multinom_naivebayes.hpp"
#include "classifier_svm.hpp"

#include "search_bif.hpp"
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

#include "util.h"
#include <fstream>

/*! \section example31 Example 31: Individual ranking (BIF) in very high-dimensional feature selection

Very high-dimensional feature selection is applied, e.g., in text categorization, with
dimensionality in the order of 10000 or 100000. Individual feature ranking (or Best Individual Feature, BIF)
is the most commonly applied approach because of its key advantages -- speed and high stability.
In this example we illustrate a less common but very effective approach based on
the Multinomial Bhattacharyya distance feature selection criterion. Multinomial Bhattacharyya has been
shown capable of overperforming traditional tools like Information Gain etc., cf. Novovicova et al.,
LNCS 4109, 2006. Randomly sampled 50% of data is used here for multinomial model parameter estimation
to be used in the actual feature selection process, another (disjunct) 40% of data is randomly sampled
for testing. The selected subset is eventually used for validation; multinomial Naive Bayes
classifier is trained on the training data on the selected subset
and classification accuracy is finally estimated on the test data.
*/

void usage(){
	std::wcout << "Usage: BIF.exe data_file sel_feat_num out_file" << std::endl;
	return;
}

int main(int argc, char** args)
{
	if (argc != 4){
		usage();
		return -1;
	}

	typedef double RETURNTYPE; 	typedef double DATATYPE;  typedef double REALTYPE;
	typedef unsigned int IDXTYPE;  typedef unsigned int DIMTYPE;  typedef short BINTYPE;
	typedef FST::Subset<BINTYPE, DIMTYPE> SUBSET;
	typedef FST::Data_Intervaller<std::vector<FST::Data_Interval<IDXTYPE> >, IDXTYPE> INTERVALLER;
	typedef boost::shared_ptr<FST::Data_Splitter<INTERVALLER, IDXTYPE> > PSPLITTER;
	typedef FST::Data_Splitter_RandomRandom<INTERVALLER, IDXTYPE, BINTYPE> SPLITTERRR;
	//typedef FST::Data_Accessor_Splitting_MemTRN<DATATYPE,IDXTYPE,INTERVALLER> DATAACCESSOR; // uncomment for TRN data format
	typedef FST::Data_Accessor_Splitting_MemARFF<DATATYPE, IDXTYPE, INTERVALLER> DATAACCESSOR; // uncomment for ARFF data format
	typedef FST::Criterion_Multinomial_Bhattacharyya<RETURNTYPE, DATATYPE, REALTYPE, IDXTYPE, DIMTYPE, SUBSET, DATAACCESSOR> BHATTMULTINOMIALDIST;
	typedef FST::Classifier_LIBSVM<RETURNTYPE, IDXTYPE, DIMTYPE, SUBSET, DATAACCESSOR> CLASSIFIERMULTINOMIAL;

	string data_file = args[1];
	IDXTYPE perTrain = 100;// int(trainNum / double(trainNum + testNum) * 100);
	IDXTYPE perTest = 100 - perTrain;
	DIMTYPE target_subsize = atoi(args[2]);;

	string out_file = args[3];

	try{
		double startTime = get_current_time();
		std::cout << "Starting Example 31: Individual ranking (BIF) in very high-dimensional feature selection..." << std::endl;
		// randomly sample 50% of data for training and randomly sample (disjunct) 40% for independent testing of final classification performance 
		PSPLITTER dsp_outer(new SPLITTERRR(1, perTrain, perTest)); // (there will be one outer randomized split only)
		// do not scale data
		boost::shared_ptr<FST::Data_Scaler<DATATYPE> > dsc(new FST::Data_Scaler_void<DATATYPE>());
		// set-up data access
		boost::shared_ptr<std::vector<PSPLITTER> > splitters(new std::vector<PSPLITTER>); splitters->push_back(dsp_outer);
		boost::shared_ptr<DATAACCESSOR> da(new DATAACCESSOR(data_file, splitters, dsc));
		da->initialize();
		// initiate access to split data parts
		da->setSplittingDepth(0); if (!da->getFirstSplit()) throw FST::fst_error("50/40 random data split failed.");
		// initiate the storage for subset to-be-selected
		boost::shared_ptr<SUBSET> sub(new SUBSET(da->getNoOfFeatures()));
		// set-up multinomial Bhattacharyya distance criterion
		boost::shared_ptr<BHATTMULTINOMIALDIST> dmultinom(new BHATTMULTINOMIALDIST);
		dmultinom->initialize(da); // (initialization = multinomial model parameter estimation on training data)
		// set-up individual feature ranking to serve as OS initialization
		FST::Search_BIF<RETURNTYPE, DIMTYPE, SUBSET, BHATTMULTINOMIALDIST> srch;
		// target subset size must be set because a) Bhattacharyya is monotonous with respect to subset size,
		// b) in very-high-dimensional problem d-optimizing search is not feasible due to search complexity
		// run the search - first find the initial subset by means of BIF, then improve it by means of OS
		std::cout << "Feature selection setup:" << std::endl << *da << std::endl << srch << std::endl << srch << std::endl << *dmultinom << std::endl << std::endl;
		RETURNTYPE critval_train, critval_test;
		srch.set_output_detail(FST::SILENT); // set FST::SILENT to disable all text output in the course of search (FST::NORMAL is default)
		if (!srch.search(target_subsize, critval_train, sub, dmultinom, std::cout)) throw FST::fst_error("Search (BIF) not finished.");
		double endTime = get_current_time();

		std::ofstream outFile(out_file.c_str(), std::ios::out);
		if (outFile){
            std::cout<<"Save result to "<<out_file<<std::endl;
			outFile << "#Train time: " << endTime - startTime << std::endl;

			outFile << *sub;
			outFile.close();
		}
		else{
			std::cerr << "open file " << out_file << " failed!" << endl;
		}
		// (optionally) validate result by estimating Naive Multinomial Bayes classifier accuracy on selected feature sub-space on independent test data
		/*
		boost::shared_ptr<CLASSIFIERMULTINOMIAL> cmultinom(new CLASSIFIERMULTINOMIAL);
		cmultinom->initialize(da);
		cmultinom->train(da, sub);
		cmultinom->test(critval_test, da);
		std::cout << "Validated Multinomial NaiveBayes accuracy=" << critval_test << std::endl << std::endl;
		*/
	}
	catch (FST::fst_error &e) { std::cerr << "FST ERROR: " << e.what() << ", code=" << e.code() << std::endl; }
	catch (std::exception &e) { std::cerr << "non-FST ERROR: " << e.what() << std::endl; }
	return 0;
}
