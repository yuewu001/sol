#ifndef HEADER_SOL_LOSS
#define HEADER_SOL_LOSS

#include "LogisticLoss.h"
#include "HingeLoss.h"
#include "SquareLoss.h"
#include "SquaredHingeLoss.h"

#include <cstring>
using std::string;

namespace SOL{
	template <typename T1, typename T2>
	LossFunction<T1,T2>* GetLossFunc(const string& str_loss) {
		string loss = str_loss;
		ToLowerCase(loss);
		if (loss == "hinge")
			return new HingeLoss<T1,T2>();
		else if (loss == "logit")
			return new LogisticLoss<T1,T2>();
		else if (loss == "square")
			return new SquareLoss<T1,T2>();
		else if (loss == "squarehinge")
			return new SquaredHingeLoss<T1, T2>();
		else{
			cerr<<"ERROR: unrecognized Loss function "<<str_loss<<endl;
			return NULL;
		}
	}
}
#endif
