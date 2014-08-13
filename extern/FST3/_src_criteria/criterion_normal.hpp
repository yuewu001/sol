#ifndef FSTCRITERIONNORMAL_H
#define FSTCRITERIONNORMAL_H

/*!======================================================================
   Feature Selection Toolbox 3 source code
   ---------------------------------------
	
   \file    criterion_normal.hpp
   \brief   Defines interface for feature selection criterion implementations based on normal model
   \author  Petr Somol (somol@utia.cas.cz) with collaborators, see Contacts at http://fst.utia.cz
   \date    October 2010
   \version 3.0.0.beta
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
#include <iostream>
#include <cmath>
#include "error.hpp"
#include "global.hpp"
#include "criterion.hpp"
#include "model_normal.hpp"

/*============== Template parameter type naming conventions ==============
--------- Numeric types: -------------------------------------------------
DATATYPE - data sample values - usually real numbers (but may be integers
          in text processing etc.)
REALTYPE - must be real numbers - for representing intermediate results of 
          calculations like mean, covariance etc.
IDXTYPE - index values for enumeration of data samples - (nonnegative) integers, 
          extent depends on numbers of samples in data
DIMTYPE - index values for enumeration of features (dimensions), or classes (not 
          class sizes) - (nonnegative) integers, usually lower extent than IDXTYPE, 
          but be aware of expressions like _classes*_features*_features ! 
          in linearized representations of feature matrices for all classes
BINTYPE - feature selection marker type - represents ca. <10 different feature 
          states (selected, deselected, sel./desel. temporarily 1st nested loop, 2nd...)
RETURNTYPE - criterion value: real value, but may be extended in future to support 
          multiple values 
--------- Class types: ---------------------------------------------------
SUBSET       - class of class type Subset 
CLASSIFIER   - class implementing interface defined in abstract class Classifier 
EVALUATOR    - class implementing interface defined in abstract class Sequential_Step 
DISTANCE     - class implementing interface defined in abstract class Distance 
DATAACCESSOR - class implementing interface defined in abstract class Data_Accessor 
INTERVALCONTAINER - class of class type TIntervaller 
CONTAINER    - STL container of class type TInterval  
========================================================================== */

namespace FST {

//! Abstract class, defines interface for feature selection criterion implementations based on normal model
template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
class Criterion_Normal : public Criterion_Filter<RETURNTYPE,SUBSET> { // abstract class
public:
	typedef boost::shared_ptr<DATAACCESSOR> PDataAccessor;
	typedef boost::shared_ptr<SUBSET> PSubset;
	Criterion_Normal();
	virtual ~Criterion_Normal() {notify("Criterion_Normal destructor.");}
	virtual bool initialize(PDataAccessor da); 

	virtual bool evaluate(RETURNTYPE &result, const PSubset sub) =0;

	DIMTYPE get_n() const {assert(_model); return _model->get_n();}
	DIMTYPE get_d() const {assert(_model); return _model->get_d();}
protected:
	Criterion_Normal(const Criterion_Normal& cn); // copy-constructor
protected:
	boost::scoped_ptr<Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR> > _model;
};

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Criterion_Normal<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Criterion_Normal(const Criterion_Normal& cn)
{
	notify("Criterion_Normal copy-constructor.");
	if(cn._model) _model.reset(new Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>(*cn._model));
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
Criterion_Normal<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::Criterion_Normal()
{
	notify("Criterion_Normal constructor.");
	_model.reset(new Model_Normal<DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>);
}

template<class RETURNTYPE, typename DATATYPE, typename REALTYPE, typename IDXTYPE, typename DIMTYPE, class SUBSET, class DATAACCESSOR>
bool Criterion_Normal<RETURNTYPE,DATATYPE,REALTYPE,IDXTYPE,DIMTYPE,SUBSET,DATAACCESSOR>::initialize(PDataAccessor da)
{
	notify("Criterion_Normal::initialize().");
	assert(da->getNoOfClasses()==2); // so far only two-class classification problems are supported
	assert(_model);
	_model->learn(da);
	return true; 
}


} // namespace
#endif // FSTCRITERIONNORMAL_H ///:~
