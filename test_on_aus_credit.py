# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 10:30:14 2020

@author: rfuchs
"""

import os 

os.chdir('C:/Users/rfuchs/Documents/GitHub/MDGMM')

from copy import deepcopy

from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder

from gower import gower_matrix
from sklearn.metrics import silhouette_score


import pandas as pd

from mdgmm import MDGMM
from init_params import dim_reduce_init
from metrics import misc
from data_preprocessing import gen_categ_as_bin_dataset, \
        compute_nj

import autograd.numpy as np


###############################################################################
######################## Credit data vizualisation    #########################
###############################################################################

#===========================================#
# Importing data
#===========================================#
os.chdir('C:/Users/rfuchs/Documents/These/Stats/mixed_dgmm/datasets')

credit = pd.read_csv('australian_credit/australian.csv', sep = ' ', header = None)
y = credit.iloc[:,:-1]
labels = credit.iloc[:,-1]

y = y.infer_objects()
numobs = len(y)


n_clusters = len(np.unique(labels))
p = y.shape[1]

#===========================================#
# Formating the data
#===========================================#
var_distrib = np.array(['categorical', 'continuous', 'continuous', 'categorical',\
                        'categorical', 'categorical', 'continuous', 'categorical',\
                        'categorical', 'continuous', 'categorical', 'categorical',\
                        'continuous', 'continuous']) 
 
# No ordinal data 
 
y_categ_non_enc = deepcopy(y)
vd_categ_non_enc = deepcopy(var_distrib)

# Encode categorical datas
y, var_distrib = gen_categ_as_bin_dataset(y, var_distrib)

# No binary data 

enc = OneHotEncoder(sparse = False, drop = 'first')
labels_oh = enc.fit_transform(np.array(labels).reshape(-1,1)).flatten()

nj, nj_bin, nj_ord = compute_nj(y, var_distrib)
y_np = y.values
nb_cont = np.sum(var_distrib == 'continuous')

p_new = y.shape[1]

# Feature category (cf)
cf_non_enc = np.logical_or(vd_categ_non_enc == 'categorical', vd_categ_non_enc == 'bernoulli')

# Non encoded version of the dataset:
y_nenc_typed = y_categ_non_enc.astype(np.object)
y_np_nenc = y_nenc_typed.values

# Defining distances over the non encoded features
dm = gower_matrix(y_nenc_typed, cat_features = cf_non_enc) 


#===========================================#
# Running the algorithm
#===========================================# 

r = {'c': [nb_cont], 'd': [3], 't': [2, 1]}
k = {'c': [1], 'd': [2], 't': [n_clusters, 1]}

seed = 1
init_seed = 2
    
eps = 1E-05
it = 15
maxstep = 100

dtype = {y.columns[j]: np.float64 if (var_distrib[j] != 'bernoulli') & \
        (var_distrib[j] != 'categorical') else np.str for j in range(p_new)}

y = y.astype(dtype, copy=True)

# MCA init
prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None)
m, pred = misc(labels_oh, prince_init['classes'], True) 
print(m)
print(confusion_matrix(labels_oh, pred))
print('Silhouette', silhouette_score(dm, pred, metric = 'precomputed'))


out = MDGMM(y_np, n_clusters, r, k, prince_init, var_distrib, nj, it, eps,\
            maxstep, seed, perform_selec = True)
m, pred = misc(labels_oh, out['classes'], True) 
print(m)
print(confusion_matrix(labels_oh, pred))
print('Silhouette', silhouette_score(dm, pred, metric = 'precomputed'))

#===========================================#
# Try auto mode
#===========================================# 

r = {'c': [nb_cont], 'd': [3], 't': [2, 1]}
k = {'c': [1], 'd': [2], 't': [5, 1]}

n_clusters = 'auto'
prince_init = dim_reduce_init(y, 'auto', k, r, nj, var_distrib, seed = None)
out = MDGMM(y_np, 'auto', r, k, prince_init, var_distrib, nj, it, eps,\
            maxstep, seed, perform_selec = True)
print('Silhouette', silhouette_score(dm, out['classes'], metric = 'precomputed'))

#===========================================#
# Final plotting
#===========================================# 

# Plot the final groups

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

colors = ['red','green']

fig = plt.figure(figsize=(8,8))
plt.scatter(out["z"][:, 0], out["z"][:, 1]  ,c=pred,\
            cmap=matplotlib.colors.ListedColormap(colors))

cb = plt.colorbar()
loc = np.arange(0,max(labels_oh), max(labels_oh)/float(len(colors)))
cb.set_ticks(loc)
cb.set_ticklabels(colors)


#=========================================================================
# Performance measure : Finding the best specification for init and DDGMM
#=========================================================================

res_folder = 'C:/Users/rfuchs/Documents/These/Experiences/mixed_algos/aus_credit'

# Init
# Best one r = ?

# Generate possible r list
max_tail_layer = 3
max_c_layer = 2
max_d_layer = 2

r_list = []

# Que les architecture minimale pour l'instant
for tl in range(2, max_tail_layer + 1):
    for cl in range(1, max_c_layer + 1):
        for dl in range(1, max_d_layer + 1):

            rc = list(range(1, nb_cont + 1))
            rc.reverse()
    
            rd = list(range(1, dl + tl + 1))
            rd.reverse()
    
            r_cdt =  {'c': rc[:cl], 'd': rd[:dl], 't': rd[dl:]}
            r_list.append(r_cdt)
            

# Generate possible k list
# Que les architecture minimale pour l'instant
k_list = []
for tl in range(2, max_tail_layer + 1):
    for cl in range(1, max_c_layer + 1):
        for dl in range(1, max_d_layer + 1):

            kc = np.random.randint(2, 5, cl)
            kd = np.random.randint(2, 5, dl)
            kt = np.random.randint(2, 5, tl)

            # Last kt must be 1 and first kc also   
            # Define the first layer as clustering layer
            kc[0] = 1
            kt[0] = n_clusters
            kt[-1] = 1
            

            k_cdt =  {'c': kc, 'd': kd, 't': kt}
            k_list.append(k_cdt)
    

numobs = len(y)
nb_trials= 30
mca_mdgmm_res = pd.DataFrame(columns = ['it_id', 'k', 'r', 'micro', 'macro',\
                                        'silhouette'])

for r, k in zip(r_list, k_list): 
    print(r)
    check_inputs(k, r)
    for i in range(nb_trials):
        # Prince init
        prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None)
        m, pred = misc(labels_oh, prince_init['classes'], True) 
        
        sil = silhouette_score(dm, pred, metric = 'precomputed')            
        micro = precision_score(labels_oh, pred, average = 'micro')
        macro = precision_score(labels_oh, pred, average = 'macro')
    
        mca_mdgmm_res = mca_mdgmm_res.append({'it_id': i + 1, 'r': str(r),\
                                              'k': k,\
                                              'micro': micro, 'macro': macro, \
                                              'silhouette': sil},\
                                               ignore_index=True)
           
mca_mdgmm_res.groupby('r').mean()
mca_mdgmm_res.groupby('r').std()

mca_mdgmm_res.to_csv(res_folder + '/mca_mdgmm_res.csv')

#============================================
# MDGMM. Thresholds use: ? and ?
#============================================

# First find the best architecture 
numobs = len(y)
r = {'c': [nb_cont], 'd': [5], 't': [4, 3]}
k = {'c': [1], 'd': [4], 't': [n_clusters, 1]}

eps = 1E-05
it = 30
maxstep = 100

nb_trials= 30
mdgmm_res = pd.DataFrame(columns = ['it_id', 'micro', 'macro', 'purity'])


prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None)
out = MDGMM(y_np, n_clusters, r, k, prince_init, var_distrib, nj, it, eps,\
            maxstep, seed = None)

r = out['best_r']
numobs = len(y)
k = out['best_k']
eps = 1E-05
it = 30
maxstep = 100

nb_trials= 30
mdgmm_res = pd.DataFrame(columns = ['it_id', 'micro', 'macro', 'silhouette'])

for i in range(nb_trials):

    print(i)
    # Prince init
    prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None)

    try:
        out = MDGMM(y_np, n_clusters, r, k, prince_init, var_distrib, nj, it,\
                    eps, maxstep, perform_selec = False, seed = None)
        m, pred = misc(labels_oh, out['classes'], True) 
        cm = confusion_matrix(labels_oh, pred)
        purity = cluster_purity(cm)

        sil = silhouette_score(dm, pred, metric = 'precomputed')                    
        micro = precision_score(labels_oh, pred, average = 'micro')
        macro = precision_score(labels_oh, pred, average = 'macro')

        mdgmm_res = mdgmm_res.append({'it_id': i + 1, 'micro': micro,\
                                    'macro': macro, 'silhouette': sil},\
                                     ignore_index=True)
    except:
        mdgmm_res = mdgmm_res.append({'it_id': i + 1, 'micro': np.nan,\
                                     'macro': np.nan, 'silhouette': np.nan},\
                                     ignore_index=True)



mdgmm_res.mean()
mdgmm_res.std()

mdgmm_res.to_csv(res_folder + '/mdgmm_res.csv')
