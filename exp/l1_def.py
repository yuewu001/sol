# define the l1 normalized parameter

SGD_list = [0, 1]
rcv1_util = [0, 1e-8,1e-7,1e-6,5e-6,
        1e-5,2.5e-5,5e-5,7.5e-5,
        1e-4, 1.25e-4,2e-4,2.5e-4,5e-4,7.5e-4, 
        1e-3]
rcv1_SSAROW = [0, 1e-8,1e-7,1e-6,5e-6,
        1e-5,2.5e-5,5e-5,7.5e-5,
        1e-4, 1.25e-4,2e-4,2.5e-4,5e-4,7.5e-4, 
        1e-3, 2e-3,3e-3,4e-3,5e-3]

MNIST_util = [0,1e-5,1e-4,1e-3,3e-3,5e-3,7e-3,7e-3,1e-2,2e-2,3e-2]

epsilon_SSAROW = [0, 1e-7,1e-6,
        1e-5,2.5e-5,5e-5,7.5e-5,
        1e-4, 2.5e-4,5e-4,7.5e-4, 
        1e-3, 1.5e-3,2e-3,2.5e-3,3e-3]
 
ASAROW_list = [0.5,0.4,0.3,0.2, 0.15,0.10, 0.05, 0.025, 0.01,0.005]

#ASAROW_lambda_list = [0.3,0.2, 0.15,0.10, 0.05, 0.025, 0.01,0.005]
#ASAROW_lambda_list = [0.10, 0.05, 0.025, 0.01,0.005]

def get_MNIST(opt_name):
    return MNIST_util
def get_rcv1(opt_name):
    if opt_name == 'SSAROW':
        return rcv1_SSAROW
    else:
        return rcv1_util
def get_news(opt_name):
    return rcv1_util

def get_lambda_list(dataset, opt_name):
    if opt_name == 'ASAROW':
        return ASAROW_list

    if dataset == 'MNIST':
        return get_MNIST(opt_name)
    else:
        return get_rcv1(opt_name)

