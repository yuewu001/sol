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
	LossFunction<T1,T2>* GetLossFunc(string loss_type) {
		ToLowerCase(loss_type);
		if (loss_type == "hinge")
			return new HingeLoss<T1,T2>();
		else if (loss_type == "logit")
			return new LogisticLoss<T1,T2>();
		else if (loss_type == "square")
			return new SquareLoss<T1,T2>();
		else if (loss_type == "squarehinge")
			return new SquaredHingeLoss<T1, T2>();
		else{
			cerr<<"ERROR: unrecognized Loss function "<<loss_type<<endl;
			return NULL;
		}
	}
}
#endif
