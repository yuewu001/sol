/*************************************************************************
	> File Name: OnlineModel.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 5/5/2014 3:24:43 PM
	> Functions: Interface for online model
 ************************************************************************/
#ifndef HEADER_ONLINE_MODEL
#define HEADER_ONLINE_MODEL

#include "../LearnModel.h"

/**
*  namespace: Batch and Online Classification
*/
namespace BOC {
    template <typename FeatType, typename LabelType> 
    class OnlineModel : public LearnModel<FeatType, LabelType> {
	
        protected:
            //iteration number
            size_t curIterNum;

            //initial learning step
            size_t initial_t;
            //power_t of the decreasing coefficient of learning rate
            float power_t;
            //initial learning rate
            float eta0; 
            //learning rate
            float eta; 

        public:
            OnlineModel(LossFunction<FeatType, LabelType> *lossFunc) : LearnModel<FeatType, LabelType>(lossFunc){
				this->modelType = "online";

                this->curIterNum = 0;

				this->initial_t = 0;
				this->power_t = 0;
				this->eta0 = 0;
            }

            virtual ~OnlineModel() {
            }

            /**
             * @Synopsis inherited functions
             */
        public:
            /**
             * PrintOptInfo print the info of optimization algorithm
             */
            virtual void PrintOptInfo() const {
                LearnModel<FeatType, LabelType>::PrintOptInfo();

                printf("Online Learning\n");
                printf("\tLearning Rate: %g\n", this->eta0);
                printf("\tInitial t  : %lu\n", this->initial_t);
                printf("\tPower t : %g\n", this->power_t);
            }

            /**
             * @Synopsis BeginTrain Reset the optimizer to the initialization status of training
             */
            virtual void BeginTrain() {
                this->curIterNum = this->initial_t;
            }

            /**
             * @Synopsis newly defined functions
             */
        public:
            /**
             * @Synopsis Iterate Iteration of online learning
             *
             * @Param x current input data example
             *
             * @Returns  prediction of the current example
             */
            virtual float Iterate(const DataPoint<FeatType, LabelType> &x) = 0;

            /**
             * @Synopsis SetParameter set the basic online learning parameters
             *
             */
			virtual void SetParameter(BOC::Params &param){
				this->initial_t = param.IntValue("-t0");
				this->power_t = param.FloatValue("-power_t");
				this->eta0 = param.FloatValue("-eta");
            }
	};

	//calculate learning rate
	inline float pEta_general(size_t t, float pt){
		return powf((float)t, pt);
	}
	inline float pEta_sqrt(size_t t, float pt){
		return sqrtf((float)t);
	}
	inline float pEta_linear(size_t t, float pt){
		return (float)t;
	}
	inline float pEta_const(size_t t, float pt){
		return 1;
	}
}

#endif

