# define the l1 normalized parameter

SGD_list = [0]
vw_list = [0]
synthetic_util = [0, 1e-8,1e-7,1e-6,5e-6,
        1e-5,2.5e-5,5e-5,7.5e-5,
        1e-4, 1.25e-4,2e-4,2.5e-4,5e-4,7.5e-4,
        1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3]
synthetic_SSAROW = [0,1e-5,2.5e-5,5e-5,7.5e-5,
        1e-4, 1.25e-4,2e-4,2.5e-4,5e-4,7.5e-4,
        1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,9.25e-3,9.5e-3,9.75e-3,1e-2]
a9a_util = [0, 1e-4, 1.25e-4,2e-4,2.5e-4,5e-4,7.5e-4,
                1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,
                1e-2,2e-2,3e-2,4e-2,5e-2,6e-2,7e-2,8e-2,9e-2,1e-1]
pcmac_util = [0, 1e-8,1e-7,1e-6,5e-6,
        1e-5,2.5e-5,5e-5,7.5e-5,
        1e-4, 1.25e-4,2e-4,2.5e-4,5e-4,7.5e-4, 
        1e-3,1.25e-3,1.5e-3,1.75e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,1e-2]

#rcv1_util = [0, 1e-8,1e-7,1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1e-1,1,10]

rcv1_util = [0, 1e-8,1e-7,1e-6,5e-6,
        1e-5,2.5e-5,5e-5,7.5e-5,
        1e-4, 1.25e-4,2e-4,2.5e-4,5e-4,7.5e-4, 
        1e-3]
rcv1_SSAROW = [0, 1e-8,1e-7,1e-6,5e-6,
        1e-5,2.5e-5,5e-5,7.5e-5,
        1e-4, 1.25e-4,2e-4,2.5e-4,5e-4,7.5e-4, 
        1e-3, 2e-3,3e-3,4e-3,5e-3]

url_util = [0,1e-8,1e-7,
        1e-6,2.5e-6,5e-6,7.5e-6,
        1e-5,2.5e-5,5e-5,7.5e-5,
        1e-4,2.5e-4,1e-4,7.5e-4, 1e-3]
url_rda = [0,1e-8,2.5e-8,5e-8,7.5e-8,
        1e-7,2.5e-7,5e-7,7.5e-7,
        1e-6,2.5e-6,5e-6,7.5e-6,
        1e-5,2.5e-5,5e-5,7.5e-5,1e-4]

webspam_trigram_util = rcv1_util
webspam_trigram_rda = [0, 1e-10,1e-9,2.5e-9,5e-9,7.5e-9,
        1e-8,1e-7,1e-6,5e-6,
        1e-5,2.5e-5,5e-5,7.5e-5,
        1e-4]

MNIST_util = [0,1e-5,1e-4,1e-3,3e-3,5e-3,7e-3,7e-3,
        1e-2,1.25e-2,1.5e-2,1.75e-2,
        2e-2, 2.5e-2,3e-2,4e-2,4.25e-2,4.5e-2,4.75e-2,
        5e-2,5.25e-2,5.5e-2,5.75e-2,6e-2,7e-2]

epsilon_SSAROW = [0, 1e-7,1e-6,
        1e-5,2.5e-5,5e-5,7.5e-5,
        1e-4, 2.5e-4,5e-4,7.5e-4, 
        1e-3, 1.5e-3,2e-3,2.5e-3,3e-3]

ASAROW_list = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2, 0.15,0.10, 0.05, 0.025, 0.01,0.005, 0.004,0.003,0.002, 0.001,
        0.0005,0.0004,0.0003,0.0002,0.0001]

#ASAROW_lambda_list = [0.3,0.2, 0.15,0.10, 0.05, 0.025, 0.01,0.005]
#ASAROW_lambda_list = [0.10, 0.05, 0.025, 0.01,0.005]

def get_MNIST(opt_name):
    return MNIST_util
def get_synthetic(opt_name):
    if opt_name == 'SSAROW':
        return synthetic_SSAROW
    else:
        return synthetic_util

def get_rcv1(opt_name):
    if opt_name == 'SSAROW':
        return rcv1_SSAROW
    else:
        return rcv1_util
def get_url(opt_name):
    if opt_name == 'RDA' or opt_name == 'Ada-RDA' or opt_name == 'CW-RDA':
        return url_rda
    else:
        return url_util

def get_news(opt_name):
    return rcv1_util
def get_webspam_trigram(opt_name):
    if opt_name == 'RDA' or opt_name == 'Ada-RDA' or opt_name == 'CW-RDA':
        return webspam_trigram_rda
    else:
        return webspam_trigram_util

def get_lambda_list(dataset, opt_name):
    if opt_name == 'ASAROW':
        return ASAROW_list
    if opt_name == 'SGD':
        return SGD_list
    if opt_name == 'vw':
        return vw_list

    if dataset == 'MNIST':
        return get_MNIST(opt_name)
    elif dataset == 'webspam_trigram':
        return get_webspam_trigram(opt_name)
    elif dataset =='synthetic':
        return get_synthetic(opt_name)
    elif dataset == 'pcmac':
        return pcmac_util
    elif dataset == 'a9a':
        return a9a_util
    elif dataset == 'url':
        return get_url(opt_name)
    else:
        return get_rcv1(opt_name)

