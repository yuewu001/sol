# define the l1 normalized parameter

SGD_list = [0]
vw_list = [0]
def get_synthetic_100_10K_100K(opt_name):
    if opt_name == 'liblinear':
        return [0.00001,0.0001,0.00025,0.0005,0.00075,0.001,0.01,0.015,0.016,0.017,0.018,0.019,0.02]
    else:
        return [50,60,70,80,90,100,120,140,160,180,200]

def get_lambda_list(dataset, opt_name):
    if dataset == 'synthetic_100_10K_100K':
        return get_synthetic_100_10K_100K(opt_name)
