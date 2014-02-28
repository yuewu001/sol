/************************************************************************
  > Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
  > Created Time: 8/19/2013 Monday 10:37:08 AM
  > Functions: Stochastic Gradient Descent
 ************************************************************************/

#ifndef HEADER_MRMR_OGD
#define HEADER_MRMR_OGD

#include "Optimizer.h"
#include "../io/basic_io.h"

#include <algorithm>

namespace SOL {
    template <typename FeatType, typename LabelType>
        class mRMR_OGD: public Optimizer<FeatType, LabelType> {
		protected:
			s_array<char> sel_feat_flag_vec;
			IndexType max_index;
            protected:
                float (*pEta_time)(size_t t, float pt);
            public:
                mRMR_OGD(DataSet<FeatType, LabelType> &dataset, LossFunction<FeatType, LabelType> &lossFunc);
                virtual ~mRMR_OGD();

            protected:
                //Reset the optimizer to the initialization status of training
                virtual void BeginTrain();
            protected:
                //this is the core of different updating algorithms
                virtual float UpdateWeightVec(const DataPoint<FeatType, LabelType> &x);

		public:
			//load the feature selection result
			bool LoadFSResult(const string& filename, int k = -1);
		};

		template <typename FeatType, typename LabelType>
		mRMR_OGD<FeatType, LabelType>::mRMR_OGD(DataSet<FeatType, LabelType> &dataset, 
			LossFunction<FeatType, LabelType> &lossFunc): Optimizer<FeatType, LabelType>(dataset, lossFunc) {
				this->id_str = "mRMR_OGD";
		}

		template <typename FeatType, typename LabelType>
		mRMR_OGD<FeatType, LabelType>::~mRMR_OGD() {
		}

		//update weight vector with stochastic gradient descent
		template <typename FeatType, typename LabelType>
		float mRMR_OGD<FeatType,LabelType>::UpdateWeightVec(
			const DataPoint<FeatType, LabelType> &x) {
				this->eta = this->eta0 / this->pEta_time(this->curIterNum, this->power_t);

				float y = this->Predict(x);
				size_t featDim = x.indexes.size();
				float gt_i = this->lossFunc->GetGradient(x.label,y);

				for (size_t i = 0; i < featDim; i++){
					if (x.indexes[i] < this->max_index && 
						this->sel_feat_flag_vec[x.indexes[i]] == 1)
						this->weightVec[x.indexes[i]] -= this->eta * gt_i * x.features[i];
				}

				//update bias 
				this->weightVec[0] -= this->eta * gt_i;

				return y;
		}
		//reset the optimizer to this initialization
		template <typename FeatType, typename LabelType>
		void mRMR_OGD<FeatType, LabelType>::BeginTrain() {
			Optimizer<FeatType, LabelType>::BeginTrain();

			if (this->power_t == 0.5)
				this->pEta_time = pEta_sqrt;
			else if(this->power_t == 0)
				this->pEta_time = pEta_const;
			else if (this->power_t == 1)
				this->pEta_time = pEta_linear;
			else
				this->pEta_time = pEta_general;
		}

		//load the feature selection result
		template <typename FeatType, typename LabelType>
		bool mRMR_OGD<FeatType, LabelType>::LoadFSResult(const string& filename, int k){
			this->max_index = 0;
			this->sel_feat_flag_vec.erase();
			basic_io io_handler;
			io_handler.open_file(filename.c_str(),"r");
			int err_code = io_handler.good();
			if (err_code  != 0){
				fprintf(stderr,"open file %s failed, error code: %d\n!", filename.c_str(), err_code);
				return false;
			}
			size_t dst_len = 64;
			char* line = new char [dst_len];
			IndexType featIndex = 0;
			while(io_handler.read_line(line,dst_len) != NULL){
				if (line[0] == '\n' || line[0] == '\r')
					continue;
				featIndex = (IndexType)(atoi(line));
				if (featIndex == 0){
					fprintf(stderr, "parse index %s failed!\n",line);
					err_code = 1;
					break;
				}
				if (this->max_index < featIndex){
					this->max_index = featIndex;
					this->sel_feat_flag_vec.reserve(featIndex + 1);
					this->sel_feat_flag_vec.resize(featIndex + 1);
					this->sel_feat_flag_vec.zeros(this->sel_feat_flag_vec.begin + max_index + 1,
						this->sel_feat_flag_vec.end);
				}
				this->sel_feat_flag_vec[featIndex] = 1;
				if (k > 0 ){					
					if (--k == 0)break;
				}
			}
			delete []line;
			return err_code == 0 ? true : false;
		}
}
#endif
