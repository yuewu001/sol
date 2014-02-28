# define the l1 normalized parameter

import sys
SGD_list = [0]
vw_list = [0]

sparse_vec = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.975, 0.99,0.995]
mrmr_vec = [50,100,150,200,250,300,350,400,450,500]
ll_vec = [0.015625,0.03125,0.0625,0.125,0.25,0.5,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,9182,18364]
def get_synthetic_100_10K_100K(opt_name):
    if opt_name == 'liblinear':
        return [0.00001,0.0001,0.00025,0.0005,0.00075,0.001,0.01,0.015,0.016,0.017,0.018,0.019,0.02]
    else:
        return [50,60,70,80,90,100,120,140,160,180,200]

def get_synthetic_200_20K_100K(opt_name):
    if opt_name == 'liblinear':
        #return [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0008,0.0007,0.0009,0.001,0.01,0.012,0.013,0.014,0.015,0.016,0.018,0.02]
        return [0.0001,0.0002,0.0003,0.0004,0.018]
    else:
        return [150,160,170,180,190,200,220,240,260,280,300,400]

def get_synthetic_500_1B_1M(opt_name):
    if opt_name == 'liblinear':
        #return [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0008,0.0007,0.0009,0.001,0.01,0.012,0.013,0.014,0.015,0.016,0.018,0.02]
        return [0.012]
    else:
        return [500]

def get_lambda_list(ds, opt_name):
    if opt_name == 'AROW' or opt_name == 'SGD':
        return [0]
    if ds == 'synthetic_100_10K_100K':
        return get_synthetic_100_10K_100K(opt_name)
    elif ds == 'synthetic_200_20K_100K':
        return get_synthetic_200_20K_100K(opt_name)
    elif ds == 'synthetic_500_1B_1M':
        return get_synthetic_500_1B_1M(opt_name)    
    elif opt_name == 'liblinear':
        return ll_vec
    elif opt_name == 'mRMR':
        return mrmr_vec
    else:
        return sparse_vec

