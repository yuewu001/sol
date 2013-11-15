# define the l1 normalized parameter

util_lambda_list = [1e-8,1e-7,1e-6,0.5e-5,1e-5,1e-4,0.5e-3,1e-3,0.5e-2,1e-2]
SSAROW_lambda_list = [1e-8,1e-7,1e-6,0.5e-5,1e-5,
        0.25e-4,0.5e-4,0.75e-4,1e-4,
        0.125e-3,0.2e-3,0.25e-3,0.5e-3,0.75e-3,1e-3,
        1e-2]
#ASAROW_lambda_list = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2, 0.15,0.10, 0.05, 0.025, 0.01,0.005]
ASAROW_lambda_list = [0.9,0.8,0.7,0.6,0.5,0.4, 0.3,0.2, 0.15,0.10, 0.05, 0.025, 0.01,0.005]
#ASAROW_lambda_list = [0.10, 0.05, 0.025, 0.01,0.005]

def get_lambda_list(opt_name):
    if opt_name == 'SSAROW':
        return SSAROW_lambda_list
    elif opt_name == 'ASAROW':
        return ASAROW_lambda_list
    elif opt_name == 'AROW':
        return SSAROW_lambda_list
    else:
        return util_lambda_list

