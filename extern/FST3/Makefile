# FST3 3.1.1.beta Makefile
###################################################
###################################################
# begin of system dependent section

## Linux settings (note: manual adjustment of Boost related paths and linkage names may be needed)
COMPILER = g++
CPUOPTFLAGS = -O2
INCLUDE = -I /usr/include/boost
THLINK = -l boost_thread

## for Cygwin switch to the following
#COMPILER = g++-4
#CPUOPTFLAGS = -O2 -march=native -funroll-loops -mfpmath=sse,387
#INCLUDE = -I /usr/include/boost
#THLINK = -l boost_thread-mt.dll

# end of system dependent section
###################################################
###################################################

# sub-directory containing shared sources
GDIR = ./_src_global
# sub-directory containing data access sources
DDIR = ./_src_dataio
# sub-directory containing feature selection criteria sources
CDIR = ./_src_criteria
# sub-directory containing feature subset search sources
SDIR = ./_src_search
# (optional) directory to store LIBSVM files, see $(LIBSVMDIR)/readme.txt
LIBSVMDIR = ./libsvm

INCLUDES = -I $(GDIR) -I $(DDIR) -I $(CDIR) -I $(SDIR) -I $(LIBSVMDIR)

# sub-directory to store object files
ODIR = ./work

H_CLASSIF = $(CDIR)/classifier.hpp $(CDIR)/classifier_knn.hpp $(CDIR)/classifier_normal_bayes.hpp $(CDIR)/classifier_multinom_naivebayes.hpp \
	$(CDIR)/distance.hpp $(CDIR)/distance_euclid.hpp $(CDIR)/distance_L1.hpp $(CDIR)/distance_Lp.hpp
H_CRIT = $(CDIR)/criterion.hpp $(CDIR)/criterion_normal.hpp $(CDIR)/criterion_normal_bhattacharyya.hpp $(CDIR)/criterion_normal_gmahalanobis.hpp \
	$(CDIR)/criterion_normal_divergence.hpp $(CDIR)/criterion_multinom.hpp $(CDIR)/criterion_multinom_bhattacharyya.hpp $(CDIR)/criterion_wrapper.hpp \
	$(CDIR)/criterion_sumofweights.hpp $(CDIR)/criterion_subsetsize.hpp $(CDIR)/criterion_negative.hpp $(CDIR)/criterion_wrapper_bias_estimate.hpp
H_DATAACCESS = $(DDIR)/data_accessor.hpp $(DDIR)/data_accessor_splitting.hpp $(DDIR)/data_accessor_splitting_mem.hpp $(DDIR)/data_accessor_splitting_memTRN.hpp \
	$(DDIR)/data_file_TRN.hpp $(DDIR)/data_accessor_splitting_memARFF.hpp $(DDIR)/data_file_ARFF.hpp $(DDIR)/data_splitter.hpp \
	$(DDIR)/data_splitter_5050.hpp $(DDIR)/data_splitter_cv.hpp $(DDIR)/data_splitter_holdout.hpp $(DDIR)/data_splitter_leave1out.hpp \
	$(DDIR)/data_splitter_resub.hpp $(DDIR)/data_splitter_randrand.hpp $(DDIR)/data_splitter_randfix.hpp $(DDIR)/data_intervaller.hpp \
	$(DDIR)/data_scaler.hpp $(DDIR)/data_scaler_void.hpp $(DDIR)/data_scaler_to01.hpp $(DDIR)/data_scaler_white.hpp
H_GLOBAL = $(GDIR)/global.hpp $(GDIR)/error.hpp $(GDIR)/clonable.hpp
H_MODEL = $(CDIR)/model.hpp $(CDIR)/model_normal.hpp $(CDIR)/model_multinom.hpp $(CDIR)/indexed_vector.hpp $(CDIR)/indexed_matrix.hpp
H_SEARCH = $(SDIR)/search.hpp $(SDIR)/seq_step.hpp $(SDIR)/seq_step_straight.hpp $(SDIR)/seq_step_hybrid.hpp $(SDIR)/seq_step_ensemble.hpp \
	$(SDIR)/search_seq.hpp $(SDIR)/search_bif.hpp $(SDIR)/search_seq_sfs.hpp $(SDIR)/search_seq_sffs.hpp $(SDIR)/search_seq_sfrs.hpp $(SDIR)/search_seq_os.hpp $(SDIR)/search_seq_dos.hpp \
	$(SDIR)/search_exhaustive.hpp $(SDIR)/branch_and_bound_predictor.hpp $(SDIR)/branch_and_bound_predictor_averaging.hpp \
	$(SDIR)/search_branch_and_bound.hpp $(SDIR)/search_branch_and_bound_basic.hpp $(SDIR)/search_branch_and_bound_improved.hpp \
	$(SDIR)/search_branch_and_bound_partial_prediction.hpp $(SDIR)/search_branch_and_bound_fast.hpp $(SDIR)/search_monte_carlo.hpp \
	$(SDIR)/stopwatch.hpp
H_RESULTS = $(SDIR)/result_tracker.hpp $(SDIR)/result_tracker_dupless.hpp $(SDIR)/result_tracker_regularizer.hpp $(SDIR)/result_tracker_stabileval.hpp \
	$(SDIR)/result_tracker_feature_stats.hpp
H_SUBSET = $(GDIR)/subset.hpp
H_THREAD = $(GDIR)/thread_pool.hpp $(SDIR)/candidate_evaluator_threaded.hpp $(SDIR)/seq_step_straight_threaded.hpp $(SDIR)/search_exhaustive_threaded.hpp \
    $(SDIR)/search_bif_threaded.hpp $(SDIR)/search_branch_and_bound_improved_threaded.hpp


##################################################
#final (optimized, non-debug) compilation targets#
##################################################

demos_all: demos demos_svm

demos: demo10 demo11 demo11t demo20 demo21 demo22 demo24 demo24t demo26 demo30 demo31 demo33 demo33t demo34 \
	demo40 demo40t demo41 demo42 demo43 demo50 demo51 demo52t demo54 demo55 demo56 demo60 demo61 demo63

demos_svm: demo12t demo23 demo25t demo32t demo35t demo53 demo62


########## BASIC SUB-OPTIMAL DEMOS ###############

demo10: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo10.of $(ODIR)/global.of 
	@echo
	@echo FST3 demo10 final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo10.of $(ODIR)/global.of -odemo10 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG

demo11: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo11.of $(ODIR)/global.of
	@echo
	@echo FST3 demo11 final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo11.of $(ODIR)/global.of -odemo11 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG
	
# Note: threaded
demo11t: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) $(H_THREAD) \
	$(ODIR)/demo11t.tof $(ODIR)/global.tof
	@echo
	@echo FST3 threaded demo11t final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo11t.tof $(ODIR)/global.tof -odemo11t $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) $(THLINK) -D NDEBUG -D FST_THREADED
	
# Note: uses external library LIBSVM, see $(LIBSVMDIR)/readme.txt
# Note: threaded
demo12t: $(H_CLASSIF) $(CDIR)/classifier_svm.hpp $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) $(H_THREAD) \
	$(ODIR)/demo12t.tof $(ODIR)/global.tof \
	$(LIBSVMDIR)/svm.of $(LIBSVMDIR)/svm.h
	@echo
	@echo FST3 threaded demo12t using external LibSVM final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo12t.tof $(ODIR)/global.tof $(LIBSVMDIR)/svm.of -odemo12t $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) $(THLINK) -D NDEBUG -D FST_THREADED


########### MORE SUB-OPTIMAL DEMOS ###############

demo20: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo20.of $(ODIR)/global.of
	@echo
	@echo FST3 demo20 final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo20.of $(ODIR)/global.of -odemo20 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG

demo21: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo21.of $(ODIR)/global.of
	@echo
	@echo FST3 demo21 final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo21.of $(ODIR)/global.of -odemo21 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG

demo22: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo22.of $(ODIR)/global.of
	@echo
	@echo FST3 demo22 final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo22.of $(ODIR)/global.of -odemo22 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG

# Note: uses external library LIBSVM, see $(LIBSVMDIR)/readme.txt
demo23: $(H_CLASSIF) $(CDIR)/classifier_svm.hpp $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo23.of $(ODIR)/global.of \
	$(LIBSVMDIR)/svm.of $(LIBSVMDIR)/svm.h
	@echo
	@echo FST3 demo23 using external LibSVM final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo23.of $(ODIR)/global.of $(LIBSVMDIR)/svm.of -odemo23 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG

demo24: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo24.of $(ODIR)/global.of
	@echo
	@echo FST3 demo24 final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo24.of $(ODIR)/global.of -odemo24 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG

# Note: threaded
demo24t: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) $(H_THREAD) \
	$(ODIR)/demo24t.tof $(ODIR)/global.tof
	@echo
	@echo FST3 threaded demo24t final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo24t.tof $(ODIR)/global.tof -odemo24t $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) $(THLINK) -D NDEBUG -D FST_THREADED

# Note: uses external library LIBSVM, see $(LIBSVMDIR)/readme.txt
# Note: threaded
demo25t: $(H_CLASSIF) $(CDIR)/classifier_svm.hpp $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) $(H_THREAD) \
	$(ODIR)/demo25t.tof $(ODIR)/global.tof \
	$(LIBSVMDIR)/svm.of $(LIBSVMDIR)/svm.h
	@echo
	@echo FST3 threaded demo25t using external LibSVM final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo25t.tof $(ODIR)/global.tof $(LIBSVMDIR)/svm.of -odemo25t $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) $(THLINK) -D NDEBUG -D FST_THREADED

demo26: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo26.of $(ODIR)/global.of
	@echo
	@echo FST3 demo26 final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo26.of $(ODIR)/global.of -odemo26 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG


######### (VERY) HIGH DIMENSIONAL DEMOS ##########

demo30: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo30.of $(ODIR)/global.of
	@echo
	@echo FST3 demo30 final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo30.of $(ODIR)/global.of -odemo30 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG

demo31: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo31.of $(ODIR)/global.of
	@echo
	@echo FST3 demo31 final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo31.of $(ODIR)/global.of -odemo31 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG

# Note: uses external library LIBSVM, see $(LIBSVMDIR)/readme.txt
# Note: threaded
demo32t: $(H_CLASSIF) $(CDIR)/classifier_svm.hpp $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) $(H_THREAD) \
	$(ODIR)/demo32t.tof $(ODIR)/global.tof \
	$(LIBSVMDIR)/svm.of $(LIBSVMDIR)/svm.h
	@echo
	@echo FST3 threaded demo32t using external LibSVM final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo32t.tof $(ODIR)/global.tof $(LIBSVMDIR)/svm.of -odemo32t $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) $(THLINK) -D NDEBUG -D FST_THREADED

demo33: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo33.of $(ODIR)/global.of
	@echo
	@echo FST3 demo33 final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo33.of $(ODIR)/global.of -odemo33 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG

# Note: threaded
demo33t: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) $(H_THREAD) \
	$(ODIR)/demo33t.tof $(ODIR)/global.tof
	@echo
	@echo FST3 threaded demo33t final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo33t.tof $(ODIR)/global.tof -odemo33t $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) $(THLINK) -D NDEBUG -D FST_THREADED

demo34: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo34.of $(ODIR)/global.of
	@echo
	@echo FST3 demo34 final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo34.of $(ODIR)/global.of -odemo34 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG

# Note: uses external library LIBSVM, see $(LIBSVMDIR)/readme.txt
# Note: threaded
demo35t: $(H_CLASSIF) $(CDIR)/classifier_svm.hpp $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) $(H_THREAD) \
	$(ODIR)/demo35t.tof $(ODIR)/global.tof \
	$(LIBSVMDIR)/svm.of $(LIBSVMDIR)/svm.h
	@echo
	@echo FST3 threaded demo35t using external LibSVM final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo35t.tof $(ODIR)/global.tof $(LIBSVMDIR)/svm.of -odemo35t $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) $(THLINK) -D NDEBUG -D FST_THREADED


############# OPTIMAL SEARCH DEMOS ###############

demo40: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo40.of $(ODIR)/global.of
	@echo
	@echo FST3 demo40 final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo40.of $(ODIR)/global.of -odemo40 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG

# Note: threaded
demo40t: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) $(H_THREAD) \
	$(ODIR)/demo40t.tof $(ODIR)/global.tof
	@echo
	@echo FST3 threaded demo40t final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo40t.tof $(ODIR)/global.tof -odemo40t $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) $(THLINK) -D NDEBUG -D FST_THREADED

demo41: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo41.of $(ODIR)/global.of
	@echo
	@echo FST3 demo41 final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo41.of $(ODIR)/global.of -odemo41 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG

demo42: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo42.of $(ODIR)/global.of
	@echo
	@echo FST3 demo42 final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo42.of $(ODIR)/global.of -odemo42 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG

demo43: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo43.of $(ODIR)/global.of
	@echo
	@echo FST3 demo43 final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo43.of $(ODIR)/global.of -odemo43 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG


############ ANTI-OVER-FITTING DEMOS #############

demo50: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo50.of $(ODIR)/global.of
	@echo
	@echo FST3 demo50 final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo50.of $(ODIR)/global.of -odemo50 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG

demo51: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo51.of $(ODIR)/global.of
	@echo
	@echo FST3 demo51 final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo51.of $(ODIR)/global.of -odemo51 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG

# Note: threaded
demo52t: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) $(H_THREAD) \
	$(ODIR)/demo52t.tof $(ODIR)/global.tof
	@echo
	@echo FST3 threaded demo52t final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo52t.tof $(ODIR)/global.tof -odemo52t $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) $(THLINK) -D NDEBUG -D FST_THREADED

# Note: uses external library LIBSVM, see $(LIBSVMDIR)/readme.txt
demo53: $(H_CLASSIF) $(CDIR)/classifier_svm.hpp $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo53.of $(ODIR)/global.of \
	$(LIBSVMDIR)/svm.of $(LIBSVMDIR)/svm.h
	@echo
	@echo FST3 demo53 using external LibSVM final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo53.of $(ODIR)/global.of $(LIBSVMDIR)/svm.of -odemo53 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG

demo54: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo54.of $(ODIR)/global.of
	@echo
	@echo FST3 demo54 final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo54.of $(ODIR)/global.of -odemo54 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG

demo55: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo55.of $(ODIR)/global.of
	@echo
	@echo FST3 demo55 final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo55.of $(ODIR)/global.of -odemo55 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG

demo56: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo56.of $(ODIR)/global.of
	@echo
	@echo FST3 demo56 final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo56.of $(ODIR)/global.of -odemo56 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG


############ COMPLEMENTARY SCENARIOS ##############

demo60: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo60.of $(ODIR)/global.of
	@echo
	@echo FST3 demo60 final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo60.of $(ODIR)/global.of -odemo60 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG

demo61: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo61.of $(ODIR)/global.of
	@echo
	@echo FST3 demo61 final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo61.of $(ODIR)/global.of -odemo61 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG

# Note: uses external library LIBSVM, see $(LIBSVMDIR)/readme.txt
demo62: $(H_CLASSIF) $(CDIR)/classifier_svm.hpp $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo62.of $(ODIR)/global.of \
	$(LIBSVMDIR)/svm.of $(LIBSVMDIR)/svm.h
	@echo
	@echo FST3 demo62 using external LibSVM final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo62.of $(ODIR)/global.of $(LIBSVMDIR)/svm.of -odemo62 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG

demo63: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo63.of $(ODIR)/global.of
	@echo
	@echo FST3 demo63 final compilation...
	@echo
	$(COMPILER) $(ODIR)/demo63.of $(ODIR)/global.of -odemo63 $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG




##################################################
#          debug compilation targets             #
##################################################

demod_all: demod demod_svm

demod: demo10d demo11d demo11td demo20d demo21d demo22d demo24d demo24td demo26d demo30d demo31d demo33d demo33td demo34d \
	demo40d demo40td demo41d demo42d demo43d demo50d demo51d demo52td demo54d demo55d demo56d demo60d demo61d demo63d

demod_svm: demo12td demo23d demo25td demo32td demo35td demo53d demo62d


########## BASIC SUB-OPTIMAL DEMOS ###############

demo10d: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo10.od $(ODIR)/global.od
	@echo
	@echo FST3 demo10 debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo10.od $(ODIR)/global.od -odemo10d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG

demo11d: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo11.od $(ODIR)/global.od
	@echo
	@echo FST3 demo11 debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo11.od $(ODIR)/global.od -odemo11d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG

# Note: threaded
demo11td: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) $(H_THREAD) \
	$(ODIR)/demo11t.tod $(ODIR)/global.tod
	@echo
	@echo FST3 threaded demo11t debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo11t.tod $(ODIR)/global.tod -odemo11td -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) $(THLINK) -D DEBUG -D FST_THREADED

# Note: uses external library LIBSVM, see $(LIBSVMDIR)/readme.txt
# Note: threaded
demo12td: $(H_CLASSIF) $(CDIR)/classifier_svm.hpp $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) $(H_THREAD) \
	$(ODIR)/demo12t.tod $(ODIR)/global.tod \
	$(LIBSVMDIR)/svm.od $(LIBSVMDIR)/svm.h
	@echo
	@echo FST3 threaded demo12t using external LibSVM debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo12t.tod $(ODIR)/global.tod $(LIBSVMDIR)/svm.od -odemo12td -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) $(THLINK) -D DEBUG -D FST_THREADED


########### MORE SUB-OPTIMAL DEMOS ###############

demo20d: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo20.od $(ODIR)/global.od
	@echo
	@echo FST3 demo20 debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo20.od $(ODIR)/global.od -odemo20d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG

demo21d: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo21.od $(ODIR)/global.od
	@echo
	@echo FST3 demo21 debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo21.od $(ODIR)/global.od -odemo21d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG

demo22d: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo22.od $(ODIR)/global.od
	@echo
	@echo FST3 demo22 debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo22.od $(ODIR)/global.od -odemo22d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG

# Note: uses external library LIBSVM, see $(LIBSVMDIR)/readme.txt
demo23d: $(H_CLASSIF) $(CDIR)/classifier_svm.hpp $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo23.od $(ODIR)/global.od \
	$(LIBSVMDIR)/svm.od $(LIBSVMDIR)/svm.h
	@echo
	@echo FST3 demo23 using external LibSVM debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo23.od $(ODIR)/global.od $(LIBSVMDIR)/svm.od -odemo23d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG

demo24d: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo24.od $(ODIR)/global.od
	@echo
	@echo FST3 demo24 debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo24.od $(ODIR)/global.od -odemo24d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG

# Note: threaded
demo24td: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) $(H_THREAD) \
	$(ODIR)/demo24t.tod $(ODIR)/global.tod
	@echo
	@echo FST3 threaded demo24t debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo24t.tod $(ODIR)/global.tod -odemo24td -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) $(THLINK) -D DEBUG -D FST_THREADED

# Note: uses external library LIBSVM, see $(LIBSVMDIR)/readme.txt
# Note: threaded
demo25td: $(H_CLASSIF) $(CDIR)/classifier_svm.hpp $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) $(H_THREAD) \
	$(ODIR)/demo25t.tod $(ODIR)/global.tod \
	$(LIBSVMDIR)/svm.od $(LIBSVMDIR)/svm.h
	@echo
	@echo FST3 threaded demo25t using external LibSVM debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo25t.tod $(ODIR)/global.tod $(LIBSVMDIR)/svm.od -odemo25td -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) $(THLINK) -D DEBUG -D FST_THREADED

demo26d: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo26.od $(ODIR)/global.od
	@echo
	@echo FST3 demo26 debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo26.od $(ODIR)/global.od -odemo26d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG


######### (VERY) HIGH DIMENSIONAL DEMOS ##########

demo30d: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo30.od $(ODIR)/global.od
	@echo
	@echo FST3 demo30 debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo30.od $(ODIR)/global.od -odemo30d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG

demo31d: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo31.od $(ODIR)/global.od
	@echo
	@echo FST3 demo31 debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo31.od $(ODIR)/global.od -odemo31d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG

# Note: uses external library LIBSVM, see $(LIBSVMDIR)/readme.txt
# Note: threaded
demo32td: $(H_CLASSIF) $(CDIR)/classifier_svm.hpp $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) $(H_THREAD) \
	$(ODIR)/demo32t.tod $(ODIR)/global.tod \
	$(LIBSVMDIR)/svm.od $(LIBSVMDIR)/svm.h
	@echo
	@echo FST3 threaded demo32t using external LibSVM debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo32t.tod $(ODIR)/global.tod $(LIBSVMDIR)/svm.od -odemo32td -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) $(THLINK) -D DEBUG -D FST_THREADED

demo33d: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo33.od $(ODIR)/global.od
	@echo
	@echo FST3 demo33 debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo33.od $(ODIR)/global.od -odemo33d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG

# Note: threaded
demo33td: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) $(H_THREAD) \
	$(ODIR)/demo33t.tod $(ODIR)/global.tod
	@echo
	@echo FST3 threaded demo33t debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo33t.tod $(ODIR)/global.tod -odemo33td -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) $(THLINK) -D DEBUG -D FST_THREADED

demo34d: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo34.od $(ODIR)/global.od
	@echo
	@echo FST3 demo34 debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo34.od $(ODIR)/global.od -odemo34d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG

# Note: uses external library LIBSVM, see $(LIBSVMDIR)/readme.txt
# Note: threaded
demo35td: $(H_CLASSIF) $(CDIR)/classifier_svm.hpp $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) $(H_THREAD) \
	$(ODIR)/demo35t.tod $(ODIR)/global.tod \
	$(LIBSVMDIR)/svm.od $(LIBSVMDIR)/svm.h
	@echo
	@echo FST3 threaded demo35t using external LibSVM debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo35t.tod $(ODIR)/global.tod $(LIBSVMDIR)/svm.od -odemo35td -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) $(THLINK) -D DEBUG -D FST_THREADED


############# OPTIMAL SEARCH DEMOS ###############

demo40d: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo40.od $(ODIR)/global.od
	@echo
	@echo FST3 demo40 debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo40.od $(ODIR)/global.od -odemo40d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG

# Note: threaded
demo40td: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) $(H_THREAD) \
	$(ODIR)/demo40t.tod $(ODIR)/global.tod
	@echo
	@echo FST3 threaded demo40t debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo40t.tod $(ODIR)/global.tod -odemo40td -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) $(THLINK) -D DEBUG -D FST_THREADED

demo41d: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo41.od $(ODIR)/global.od
	@echo
	@echo FST3 demo41 debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo41.od $(ODIR)/global.od -odemo41d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG

demo42d: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo42.od $(ODIR)/global.od
	@echo
	@echo FST3 demo42 debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo42.od $(ODIR)/global.od -odemo42d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG

demo43d: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo43.od $(ODIR)/global.od
	@echo
	@echo FST3 demo43 debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo43.od $(ODIR)/global.od -odemo43d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG


############ ANTI-OVER-FITTING DEMOS #############

demo50d: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo50.od $(ODIR)/global.od
	@echo
	@echo FST3 demo50 debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo50.od $(ODIR)/global.od -odemo50d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG

demo51d: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo51.od $(ODIR)/global.od
	@echo
	@echo FST3 demo51 debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo51.od $(ODIR)/global.od -odemo51d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG

# Note: threaded
demo52td: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) $(H_THREAD) \
	$(ODIR)/demo52t.tod $(ODIR)/global.tod
	@echo
	@echo FST3 threaded demo52t debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo52t.tod $(ODIR)/global.tod -odemo52td -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) $(THLINK) -D DEBUG -D FST_THREADED

# Note: uses external library LIBSVM, see $(LIBSVMDIR)/readme.txt
demo53d: $(H_CLASSIF) $(CDIR)/classifier_svm.hpp $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo53.od $(ODIR)/global.od \
	$(LIBSVMDIR)/svm.od $(LIBSVMDIR)/svm.h
	@echo
	@echo FST3 demo53 using external LibSVM debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo53.od $(ODIR)/global.od $(LIBSVMDIR)/svm.od -odemo53d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG

demo54d: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo54.od $(ODIR)/global.od
	@echo
	@echo FST3 demo54 debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo54.od $(ODIR)/global.od -odemo54d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG

demo55d: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo55.od $(ODIR)/global.od
	@echo
	@echo FST3 demo55 debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo55.od $(ODIR)/global.od -odemo55d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG

demo56d: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo56.od $(ODIR)/global.od
	@echo
	@echo FST3 demo56 debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo56.od $(ODIR)/global.od -odemo56d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG


############ COMPLEMENTARY SCENARIOS ##############

demo60d: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo60.od $(ODIR)/global.od
	@echo
	@echo FST3 demo60 debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo60.od $(ODIR)/global.od -odemo60d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG

demo61d: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo61.od $(ODIR)/global.od
	@echo
	@echo FST3 demo61 debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo61.od $(ODIR)/global.od -odemo61d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG

# Note: uses external library LIBSVM, see $(LIBSVMDIR)/readme.txt
demo62d: $(H_CLASSIF) $(CDIR)/classifier_svm.hpp $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo62.od $(ODIR)/global.od \
	$(LIBSVMDIR)/svm.od $(LIBSVMDIR)/svm.h
	@echo
	@echo FST3 demo62 using external LibSVM debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo62.od $(ODIR)/global.od $(LIBSVMDIR)/svm.od -odemo62d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG

demo63d: $(H_CLASSIF) $(H_CRIT) $(H_DATAACCESS) $(H_GLOBAL) $(H_MODEL) $(H_SEARCH) $(H_RESULTS) $(H_SUBSET) \
	$(ODIR)/demo63.od $(ODIR)/global.od
	@echo
	@echo FST3 demo63 debug compilation...
	@echo
	$(COMPILER) -g $(ODIR)/demo63.od $(ODIR)/global.od -odemo63d -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG




##################################################
#              supporting targets                #
##################################################

clean:
	rm -rf $(ODIR)/*.of $(ODIR)/*.od $(ODIR)/*.tod $(ODIR)/*.tof
	rm -rf $(LIBSVMDIR)/*.of $(LIBSVMDIR)/*.od
	rm -rf *.o

$(ODIR)/%.of: %.cpp
	$(COMPILER) -c -o $(ODIR)/$*.of $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG $<

$(ODIR)/%.of: $(GDIR)/%.cpp
	$(COMPILER) -c -o $(ODIR)/$*.of $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG $<

$(ODIR)/%.od: %.cpp
	$(COMPILER) -g -c -o $(ODIR)/$*.od -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG $<

$(ODIR)/%.od: $(GDIR)/%.cpp
	$(COMPILER) -g -c -o $(ODIR)/$*.od -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG $<

$(ODIR)/%.tof: %.cpp
	$(COMPILER) -c -o $(ODIR)/$*.tof $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG -D FST_THREADED $<

$(ODIR)/%.tof: $(GDIR)/%.cpp
	$(COMPILER) -c -o $(ODIR)/$*.tof $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG -D FST_THREADED $<

$(ODIR)/%.tod: %.cpp
	$(COMPILER) -g -c -o $(ODIR)/$*.tod -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG -D FST_THREADED $<

$(ODIR)/%.tod: $(GDIR)/%.cpp
	$(COMPILER) -g -c -o $(ODIR)/$*.tod -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG -D FST_THREADED $<

$(LIBSVMDIR)/%.of: $(LIBSVMDIR)/%.cpp
	$(COMPILER) -c -o $(LIBSVMDIR)/$*.of $(CPUOPTFLAGS) $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D NDEBUG $<

$(LIBSVMDIR)/%.od: $(LIBSVMDIR)/%.cpp
	$(COMPILER) -g -c -o $(LIBSVMDIR)/$*.od -Wall $(INCLUDE) $(INCLUDES) $(LDFLAGS) -D DEBUG $<
