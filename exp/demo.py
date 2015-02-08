#!/usr/bin/env python
import demo_util

#model list
model_list = ['mRMR','BIF']
model_list = ['FGM','liblinear']
model_list = ['SGD','DAROW','SOFS']
model_list = ['CW_TG','CW_RDA','STG','FOBOS','RDA','Ada_FOBOS','Ada_RDA']
model_list = ['CW_TG']

#dataset list
ds_list = ['url']

#extra command sent to SOL
model_config = {
'type':'sol',
#'sol_type': 'run_all',
'sol_type': 'search_l1',
'tolerance': 0.01,
'cache':True,
'norm':False,
'bc_loss':'Hinge',
'mc_loss':'MaxScoreHinge',
'rand_num':1,
'passes':1,
'is_default_param':  False
}

demo_util.demo(ds_list, model_list, model_config)
