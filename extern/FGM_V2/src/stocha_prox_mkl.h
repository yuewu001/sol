
#ifndef _STOCHASTIC_StochProxMKL_H_
#define _STOCHASTIC_StochProxMKL_H_
#include <malloc.h>
#include <math.h>
#include <time.h>
#include "linear.h"
#include <stdlib.h>


class StochProxMKL
{
public:
	StochProxMKL();
	StochProxMKL(problem *&prob_,const parameter *svm_param_,int num_kernel_,model *&svm_model_)
	{
		num_kernel	= num_kernel_;
		svm_param 	= svm_param_; 
		prob		= prob_;
		bias        = prob->bias;
		svm_model	= svm_model_;
		l			= prob_->l; 
		w_size		= num_kernel*svm_param->B;
		max_num_kernel = svm_param->max_iteraion;
		C           = svm_param->C;
		max_w_size	= max_num_kernel*svm_param->B;

		numEpochs = 10;
		eta0      = 1.0;
		L_f       = 10;
		TAU_K     = 0.01;
		work_set_size = 1000;
		StochProxMKL_allocate();
	}
	inline static void veccpy(double *y, const double *x, const int n)
	{
		int i;

		for (i = 0;i < n;++i) {
			y[i] = x[i];
		}
	}
	~StochProxMKL();
	int StochProxMKL::proximity_L1squared(int d, double* x0, double lambda, double* x);
	int StochProxMKL::compare (const void * a, const void * b);
	//int StochProxMKL::mkl_online_linear();
	void StochProxMKL::StochProxMKL_allocate();
    void StochProxMKL::sort_rs(doubleIntPair* x0abs,int n_rsi);
	void StochProxMKL::warm_start_model(int num_kernel_);
	void StochProxMKL::smoothmkl_init();
	void StochProxMKL::calculate_alpha();
	void StochProxMKL::return_model();
	void StochProxMKL::mkl_svm_train();

    int StochProxMKL::mkl_online_linear_l1svm();
	int StochProxMKL::mkl_online_linear_l2svm();
	int StochProxMKL::mkl_batch_linear_l2svm();
	int StochProxMKL::mkl_prox_linear_l2svm();
	void StochProxMKL::calucate_gwx(double *gw);
	int StochProxMKL::stoch_mkl_prox_linear_l2svm();
    int StochProxMKL::stoch_mkl_prox_linear_lr();
	
	void StochProxMKL::calculate_wx_gwx(double s);
	double StochProxMKL::calculate_gradient(double *gw,double *w);
    double StochProxMKL::line_search(double *v, double *gw, 
							double *normsq, double *normsq_search,	double tau, double *b0,double *b, double *upd_t, double fv,double lambda,double &f_w_);
	double StochProxMKL::line_search_complex(double *v, double *gw, 
							double *normsq,	double *normsq_search,double tau, double *b0,double *b, double *upd_t, double fv,double lambda);

	int StochProxMKL::mkl_online_linear_logistic();
	double StochProxMKL::calculate_gradient_logistic(double *gw, double *v);
	int StochProxMKL::mkl_prox_linear_logistic();
	double StochProxMKL::line_search_logistic(double *v, double *gw, 
							double *normsq,	double *normsq_search,double tau, double *b0,double *b, double *upd_t, double fv,double lambda,double &f_w_);

	double StochProxMKL::line_search_logistic(double *v, double *gw, 
							double *normsq,	double *normsq_search,double tau, double *b0,double *b, double *upd_t, double fv,double lambda,double &f_w_, int *work_set);

	double StochProxMKL::calculate_gradient(double *gw, double *v, int *work_set);
    double StochProxMKL::line_search(double *v, double *gw, double *normsq, double *normsq_search,	
		 double tau, double *b0,double *b, double *upd_t, double fv,double lambda,double &f_w_, int *work_set);
	void StochProxMKL::calculate_wx_gwx(double s, int *work_set);
	void StochProxMKL::calucate_gwx(double *gw,int *work_set);
	double StochProxMKL::calculate_gradient_logistic(double *gw, double *v,int *work_set);
	protected:
	const parameter		*svm_param;
	problem				*prob;
	model               *svm_model;

	int                 l;
	int					max_iter;
	int					max_inner_iter;
	double				innereps;
	double				innereps_min;
	//double              upper_bound[3];
	double				Cp;
	double				Cn;
	double				eps;
	double				bias;
	double              ptr_fx; //objective value



	double				*grad_alpha;//gradient for each kernel:h_'(z)
	double				**QD;       //Q_ii for each sub_kernel
	double				*betas;
	double				*gwTgw;//qt+h_t(z)
	double              *wTgw;
	double				*w;
	double				*gw;
	double				*G;
	double				*v_tau;
	double              *v;
	double              *w_temp;
	int					*index;
	int				    *y;
	double				*alpha_sum;
	int					num_kernel;//number of kernels
	int					w_size;


	//for functions
	double				*lambda_f; //for objective
	double				*fval_f; //qt+h_t(z)

	//other parameters
	double				beta;
	double				*bt;
	double              mu;
	double              q; //q = 1/mu

	double				epsf;
	int					max_num_kernel;
	int					max_w_size;
	double              C2; //to control the noise 
	double              C;
	double              gp;
	double              eta_mu; //eta_ = 0.8;
	double              upper_bound[3];
	double              **wtTxt;
	double              **gw_wtTxt;
	double              **wx_gwx;
	int                 numEpochs;
	double              eta0;
	double				yita; 
	double              L_f;
	double              TAU_K;
	double              tau_k;
	//for stochastic solver
	int                 *work_set;//store the dataset
	int                 work_set_size;
	
};



#endif