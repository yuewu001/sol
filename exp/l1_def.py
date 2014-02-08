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
synthetic2_util = [0,1e-5,2e-5,3e-5,4e-5,5e-5,6e-5,7e-5,8e-5,9e-5,
	1e-4,2e-4,2.5e-4,2.75e-4,3e-4,3.5e-4,4e-4,4.5e-4,5e-4,7.5e-4, 
    1e-3,5e-3,1e-2,2e-2,2.1e-2,2.2e-2,2.3e-2,2.4e-2,2.5e-2,2.6e-2,2.7e-2,2.8e-2,2.9e-2,
	3e-2,3.1e-2]
synthetic2_SSAROW = [0,1e-4,1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,
	1e-2,2.25e-2,2.5e-2,2.75e-2,2.9e-2,2.95e-2,2.975e-2,2.985e-2,2.995e-2,
	3e-2,3.025e-2,3.03e-2,3.04e-2,3.05e-2,3.06e-2,3.07e-2,3.08e-2,3.09e-2,3.1e-2]
		
########a9a
a9a_util = [0, 1e-4, 1.25e-4,2e-4,2.5e-4,5e-4,7.5e-4,
                1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,
                1e-2,2e-2,3e-2,4e-2,5e-2,6e-2,7e-2,8e-2,9e-2,1e-1]
a9a_SSAROW= [0, 1e-4, 1.25e-4,2e-4,2.5e-4,5e-4,7.5e-4,
                1e-3,5e-3, 1e-2,2e-2,3e-2,4e-2,5e-2,6e-2,7e-2,8e-2,9e-2,
                1e-1, 2e-1,3e-1,4e-1,5e-1]
a9a_RDA = [0, 1e-4, 1.25e-4,2e-4,2.5e-4,5e-4,7.5e-4,
        1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,
        1e-2,1.25e-2,1.5e-2,1.75e-2,2e-2,2.25e-2,2.5e-2,2.75e-2, 3e-2,4e-2,5e-2]
a9a_AROW_DA = [0, 1e-4, 1.25e-4,2e-4,2.5e-4,5e-4,7.5e-4,
        1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,
        1e-2,2e-2,2.25e-2,2.5e-2,2.75e-2, 3e-2,4e-2,5e-2,6e-2,7e-2,
        1e-1,1.25e-1,1.5e-1,1.75e-1,2e-1]
def get_a9a(opt_name):
    if opt_name == 'AROW-DA':
        return a9a_AROW_DA;
    elif opt_name == 'AROW-TG':
        return a9a_SSAROW
    elif 'RDA' in opt_name:
        return a9a_RDA
    else:
        return a9a_util

pcmac_util = [0, 1e-8,1e-7,1e-6,5e-6,
        1e-5,2.5e-5,5e-5,7.5e-5,
        1e-4, 1.25e-4,2e-4,2.5e-4,5e-4,7.5e-4, 
        1e-3,1.25e-3,1.5e-3,1.75e-3,2e-3,3e-3,4e-3,5e-3,6e-3,7e-3,8e-3,9e-3,1e-2,2e-2,3e-2,4e-2,5e-2]

rcv1_util = [0, 1e-8,1e-7,1e-6,5e-6,
        1e-5,2.5e-5,5e-5,7.5e-5,
        1e-4, 1.25e-4,2e-4,2.5e-4,5e-4,7.5e-4, 
        1e-3,2e-3,3e-3,4e-3,5e-3]
rcv1_SSAROW = [0, 1e-8,1e-7,1e-6,5e-6,
        1e-5,2.5e-5,5e-5,7.5e-5,
        1e-4, 1.25e-4,2e-4,2.5e-4,5e-4,7.5e-4, 
        1e-3, 2e-3,3e-3,4e-3,5e-3]

url_util = [0,1e-8,1e-7,
        1e-6,2.5e-6,5e-6,7.5e-6,
        1e-5,2.5e-5,5e-5,7.5e-5,
        1e-4,2.5e-4,1e-4,7.5e-4, 1e-3,2e-3,3e-3,4e-3,5e-3]
url_rda = [0,1e-8,2.5e-8,5e-8,7.5e-8,
        1e-7,2.5e-7,5e-7,7.5e-7,
        1e-6,2.5e-6,5e-6,7.5e-6,
        1e-5,2.5e-5,5e-5,7.5e-5,1e-4,2.5e-4,5e-4,7.5e-4,1e-3]

webspam_trigram_util = rcv1_util
webspam_trigram_rda = [0, 1e-10,1e-9,2.5e-9,5e-9,7.5e-9,
        1e-8,1e-7,1e-6,5e-6,7.5e-6,
        1e-5,2.5e-5,5e-5,7.5e-5,
        1e-4]
########### MNIST
MNIST_util = [0,1e-5,1e-4,1e-3,3e-3,5e-3,7e-3,7e-3,
        1e-2,1.5e-2,2e-2, 2.5e-2,3e-2,4e-2,5e-2,6e-2,7e-2,8e-2,9e-2,
        1e-1,2e-1,2.25e-1,2.5e-1,2.75e-1,3e-1,4e-1,5e-1,6e-1,7e-1,8e-1,9e-1]
MNIST_DA = [0,1e-5,1e-4,1e-3,3e-3,5e-3,7e-3,7e-3,
        1e-2,1.25e-2,1.5e-2,1.75e-2,
        2e-2, 2.5e-2,3e-2,4e-2,5e-2,6e-2,7e-2,8e-2,9e-2,
        1e-1,2e-1,3e-1,4e-1]
def get_MNIST(opt_name):
    if 'DA' in opt_name:
        return MNIST_DA
    else:
        return MNIST_util

epsilon_SSAROW = [0, 1e-7,1e-6,
        1e-5,2.5e-5,5e-5,7.5e-5,
        1e-4, 2.5e-4,5e-4,7.5e-4, 
        1e-3, 1.5e-3,2e-3,2.5e-3,3e-3]

ASAROW_list = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2, 0.15,0.10, 0.05, 0.025, 0.01,0.005, 0.004,0.003,0.002, 0.001,
        0.0005,0.0004,0.0003,0.0002,0.0001]
synthetic_ASAROW_list = [0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2, 0.15,0.10, 0.05, 0.025, 0.01,0.005]
#ASAROW_lambda_list = [0.3,0.2, 0.15,0.10, 0.05, 0.025, 0.01,0.005]
#ASAROW_lambda_list = [0.10, 0.05, 0.025, 0.01,0.005]

def get_synthetic(opt_name):
    if opt_name == 'AROW-TG':
        return synthetic_SSAROW
    else:
        return synthetic_util
def get_synthetic2(opt_name):
    if opt_name == 'AROW-TG':
        return synthetic2_SSAROW
    else:
        return synthetic2_util
def get_rcv1(opt_name):
    if opt_name == 'AROW-TG':
        return rcv1_SSAROW
    else:
        return rcv1_util
def get_url(opt_name):
    if opt_name == 'RDA' or opt_name == 'Ada-RDA' or opt_name == 'AROW-DA':
        return url_rda
    else:
        return url_util

def get_news(opt_name):
    return rcv1_util
def get_webspam_trigram(opt_name):
    if opt_name == 'RDA' or opt_name == 'Ada-RDA' or opt_name == 'AROW-DA':
        return webspam_trigram_rda
    else:
        return webspam_trigram_util

def get_lambda_list(dataset, opt_name):
    if opt_name == 'AROW-FS' or opt_name == 'SGD-FS' or opt_name == 'OFSGD':
        if dataset == 'synthetic2':
	    return synthetic_ASAROW_list
        else:
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
    elif dataset =='synthetic2':
        return get_synthetic2(opt_name)
    elif dataset == 'pcmac':
        return pcmac_util
    elif dataset == 'a9a':
        return get_a9a(opt_name);
    elif dataset == 'url':
        return get_url(opt_name)
    else:
        return get_rcv1(opt_name)

