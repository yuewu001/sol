# define the l1 normalized parameter

SGD_lambda_list= [0, 1]
util_lambda_list = [0, 1e-8,1e-7,1e-6,5e-6,
        1e-5,2.5e-5,5e-5,7.5e-5,1e-4,
        1.25e-4,2e-4,2.5e-4,5e-4,7.5e-4, 1e-3]
#for rcv1
SSAROW_lambda_list = util_lambda_list;
#for epsilon
#SSAROW_lambda_list = [0, 1e-7,1e-6,
#        1e-5,2.5e-5,5e-5,7.5e-5,1e-4,
#        2.5e-4,5e-4,7.5e-4, 1e-3, 1.5e-3,2e-3,2.5e-3,3e-3]
 
AROW_lambda_list = [0, 1e-8,1e-7,1e-6,2.5e-6,
        5e-6,7.5e-6, 1e-5,1.5e-5, 2e-5, 
        2.5e-5,3.5e-5, 5e-5,7.5e-5,1e-4]

#ASAROW_lambda_list = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2, 0.15,0.10, 0.05, 0.025, 0.01,0.005]
ASAROW_lambda_list = [0.9,0.8,0.7,0.6,0.5,0.4, 0.3,0.2, 0.15,0.10, 0.05, 0.025, 0.01,0.005]
#ASAROW_lambda_list = [0.10, 0.05, 0.025, 0.01,0.005]

def get_lambda_list(opt_name):
    if opt_name == 'SSAROW':
        return SSAROW_lambda_list
    elif opt_name == 'ASAROW':
        return ASAROW_lambda_list
    elif opt_name == 'AROW':
        return AROW_lambda_list
    else:
        return util_lambda_list

