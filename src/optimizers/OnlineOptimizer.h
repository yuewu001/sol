/*************************************************************************
	> File Name: OnlineOptimizer.h
	> Copyright (C) 2013 Yue Wu<yuewu@outlook.com>
	> Created Time: 5/5/2014 6:24:15 PM
	> Functions: optimizer for online learning
	************************************************************************/
#ifndef HEADER_ONLINE_OPTIMIZER
#define HEADER_ONLINE_OPTIMIZER

#include "Optimizer.h"
#include "../algorithms/om/OnlineModel.h"

/**
*  namespace: Batch and Online Classification
*/
namespace BOC {
	template <typename FeatType, typename LabelType>
	class OnlineOptimizer : public Optimizer < FeatType, LabelType > {
		//dynamic bindings
		DECLARE_CLASS

	protected:
		/**
		 * @Synopsis Constructors
		 */
	public:
		OnlineOptimizer(OnlineModel<FeatType, LabelType> *model, DataSet<FeatType, LabelType> *dataset) :
			Optimizer<FeatType, LabelType>(model, dataset){
		}

		virtual ~OnlineOptimizer() {
		}

		/**
		 * @Synopsis inherited functions
		 */
	public:
		//train the data
		virtual float Train() {
			//reset
			if (this->Reset() == false)
				return 1.f;
			OnlineModel<FeatType, LabelType> *p_onlineModel = static_cast<OnlineModel<FeatType, LabelType>*>(this->learnModel);
			p_onlineModel->BeginTrain();
			float errorNum(0);
			size_t show_step = 1; //show information every show_step
			size_t show_count = 2;
			size_t data_count = 0;

			//double train_time  = 0;
			printf("\nIterations:\n");
			printf("\nIterate No.\t\tError Rate\t\t\n");
			while (1) {
				DataChunk<PointType> &chunk = this->dataSet->GetChunk();
				//double time1 = get_current_time();
				//all the data has been processed!
				if (chunk.dataNum == 0) {
					this->dataSet->FinishRead();
					break;
				}

				for (size_t i = 0; i < chunk.dataNum; i++) {
					PointType &data = chunk.data[i];

					p_onlineModel->UpdateModelDimention(data.dim());
					float y = p_onlineModel->Iterate(data);
					//loss
					if (p_onlineModel->IsCorrect(data.label, y) == false){
						errorNum++;
						data.margin = y * data.label;
					}
					data_count++;
					this->update_times++;
					if (show_count == data_count){
						printf("%lu\t\t\t%.6f\t\t\n", data_count,
							errorNum / (float)(data_count));
						show_count = (size_t(1) << ++show_step);
					}
				}
				//double time2 = get_current_time();
				//train_time += time2 - time1;
				this->dataSet->FinishRead();
			}
			p_onlineModel->EndTrain();
			//cout<<"Purely Training Time: "<<train_time<<" s"<<endl;
			return errorNum / this->update_times;
		}
	};

	template <typename FeatType, typename LabelType>
	ClassInfo OnlineOptimizer<FeatType, LabelType>::classInfo("opt_online",
		"optimizer for online learning models", OnlineOptimizer<FeatType, LabelType>::CreateObject);

	template <typename FeatType, typename LabelType>
	void* OnlineOptimizer<FeatType, LabelType>::CreateObject(void* model, void* dataset, void* param3) {
		return new OnlineOptimizer<FeatType, LabelType>((OnlineModel<FeatType, LabelType>*)model,
			(DataSet<FeatType, LabelType>*)dataset);
	}
}

#endif

