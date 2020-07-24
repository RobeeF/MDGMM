# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:25:11 2020

@author: rfuchs
"""

import os 

os.chdir('C:/Users/rfuchs/Documents/GitHub/MDGMM')

from copy import deepcopy

from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder

import pandas as pd

from mdgmm import MDGMM
from init_params import dim_reduce_init
from metrics import misc, cluster_purity
from data_preprocessing import gen_categ_as_bin_dataset, \
        compute_nj

import autograd.numpy as np
from autograd.numpy.random import uniform


###############################################################################
###############         Heart    vizualisation          #######################
###############################################################################

#===========================================#
# Importing data
#===========================================#
os.chdir('C:/Users/rfuchs/Documents/These/Stats/mixed_dgmm/datasets')

pima = pd.read_csv('pima/pima_indians.csv', sep = ',')
y = pima.iloc[:,:-1]
labels = pima.iloc[:,-1]

y = y.infer_objects()
numobs = len(y)


n_clusters = len(np.unique(labels))
p = y.shape[1]

#===========================================#
# Formating the data
#===========================================#
var_distrib = np.array(['ordinal', 'continuous', 'continuous', 'continuous',\
                        'continuous', 'continuous', 'continuous', 'continuous']) 
 
# Ordinal data already encoded
 
y_categ_non_enc = deepcopy(y)
vd_categ_non_enc = deepcopy(var_distrib)

# No categ data
# No binary data 

enc = OneHotEncoder(sparse = False, drop = 'first')
labels_oh = enc.fit_transform(np.array(labels).reshape(-1,1)).flatten()

nj, nj_bin, nj_ord = compute_nj(y, var_distrib)
y_np = y.values
nb_cont = np.sum(var_distrib == 'continuous')
     
#===========================================#
# Running the algorithm
#===========================================# 

r = {'c': [nb_cont], 'd': [3], 't': [2, 1]}
k = {'c': [2], 'd': [2], 't': [n_clusters,  1]}

seed = 1
init_seed = 2
    
eps = 1E-05
it = 15
maxstep = 100

# MCA init
prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None)
m, pred = misc(labels_oh, prince_init['classes'], True) 
print(m)
print(confusion_matrix(labels_oh, pred))

#init = prince_init
#y = y_np

#import warnings
#warnings.simplefilter('ignore')

out = MDGMM(y_np, n_clusters, r, k, prince_init, var_distrib, nj, it, eps, maxstep, seed)
m, pred = misc(labels_oh, classes, True) 
print(m)
print(confusion_matrix(labels_oh, pred))

# Plot the final groups

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

colors = ['red','green']

fig = plt.figure(figsize=(8,8))
plt.scatter(out["z"][:, 0], out["z"][:, 1]  ,c=labels_oh,\
            cmap=matplotlib.colors.ListedColormap(colors))

cb = plt.colorbar()
loc = np.arange(0,max(labels_oh),max(labels_oh)/float(len(colors)))
cb.set_ticks(loc)
cb.set_ticklabels(colors)


# FAMD init
famd_init = dim_reduce_init(y_categ_non_enc.infer_objects(), n_clusters,\
                    k, r, nj, vd_categ_non_enc, use_famd = True, seed = None)
m, pred = misc(labels_oh, famd_init['classes'], True) 
print(m)
print(confusion_matrix(labels_oh, pred))


#=========================================================================
# Performance measure : Finding the best specification for init and DDGMM
#=========================================================================

res_folder = 'C:/Users/rfuchs/Documents/These/Experiences/mixed_algos/breast'


# Init
# Best one r = (2,1)
numobs = len(y)
k = [n_clusters]

nb_trials= 30
mca_res = pd.DataFrame(columns = ['it_id', 'r', 'micro', 'macro', 'purity'])

for r1 in range(2, 9):
    print(r1)
    r = np.array([r1, 1])
    for i in range(nb_trials):
        # Prince init
        prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None)
        m, pred = misc(labels_oh, prince_init['classes'], True) 
        cm = confusion_matrix(labels_oh, pred)
        purity = cluster_purity(cm)
            
        micro = precision_score(labels_oh, pred, average = 'micro')
        macro = precision_score(labels_oh, pred, average = 'macro')
        #print(micro)
        #print(macro)
    
        mca_res = mca_res.append({'it_id': i + 1, 'r': str(r), 'micro': micro, 'macro': macro, \
                                        'purity': purity}, ignore_index=True)
       

mca_res.groupby('r').mean()
mca_res.groupby('r').std()

mca_res.to_csv(res_folder + '/mca_res.csv')

# DDGMM. Thresholds use: 0.5 and 0.10
r = np.array([5, 4, 2])
numobs = len(y)
k = [4, n_clusters]
eps = 1E-05
it = 30
maxstep = 100

nb_trials= 30
ddgmm_res = pd.DataFrame(columns = ['it_id', 'micro', 'macro', 'purity'])



# First fing the best architecture 
prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None)
out = DDGMM(y_np, n_clusters, r, k, prince_init, var_distrib, nj, it, eps, maxstep, seed = None)

r = out['best_r']
numobs = len(y)
k = out['best_k']
eps = 1E-05
it = 30
maxstep = 100

nb_trials= 30
ddgmm_res = pd.DataFrame(columns = ['it_id', 'micro', 'macro', 'purity'])

for i in range(nb_trials):

    print(i)
    # Prince init
    prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None)

    try:
        out = DDGMM(y_np, n_clusters, r, k, prince_init, var_distrib, nj, M, it, eps, maxstep, seed = None)
        m, pred = misc(labels_oh, out['classes'], True) 
        cm = confusion_matrix(labels_oh, pred)
        purity = cluster_purity(cm)
        
        micro = precision_score(labels_oh, pred, average = 'micro')
        macro = precision_score(labels_oh, pred, average = 'macro')
        print(micro)
        print(macro)

        ddgmm_res = ddgmm_res.append({'it_id': i + 1, 'micro': micro, 'macro': macro, \
                                    'purity': purity}, ignore_index=True)
    except:
        ddgmm_res = ddgmm_res.append({'it_id': i + 1, 'micro': np.nan, 'macro': np.nan, \
                                    'purity': np.nan}, ignore_index=True)



ddgmm_res.mean()
ddgmm_res.std()

ddgmm_res.to_csv(res_folder + '/ddgmm_res.csv')


#=======================================================================
# Performance measure : Finding the best specification for other algos
#=======================================================================

from gower import gower_matrix
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import AgglomerativeClustering
from minisom import MiniSom   
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Feature category (cf)
cf_non_enc = (vd_categ_non_enc != 'ordinal') & (vd_categ_non_enc != 'binomial')

# Non encoded version of the dataset:
y_nenc_typed = y_categ_non_enc.astype(np.object)
y_np_nenc = y_nenc_typed.values

# Defining distances over the non encoded features
dm = gower_matrix(y_nenc_typed, cat_features = cf_non_enc) 

# <nb_trials> tries for each specification
nb_trials = 30

res_folder = 'C:/Users/rfuchs/Documents/These/Experiences/mixed_algos/breast'


#****************************
# Partitional algorithm
#****************************

part_res_modes = pd.DataFrame(columns = ['it_id', 'init', 'micro', 'macro', 'purity'])

inits = ['Huang', 'Cao', 'random']

for init in inits:
    print(init)
    for i in range(nb_trials):
        km = KModes(n_clusters= n_clusters, init=init, n_init=10, verbose=0)
        kmo_labels = km.fit_predict(y_np_nenc)
        m, pred = misc(labels_oh, kmo_labels, True) 
        micro = precision_score(labels_oh, pred, average = 'micro')
        macro = precision_score(labels_oh, pred, average = 'macro')
        cm = confusion_matrix(labels_oh, pred)
        purity = cluster_purity(cm)

        part_res_modes = part_res_modes.append({'it_id': i + 1, 'init': init, \
                            'micro': micro, 'macro': macro, 'purity': purity}, \
                                               ignore_index=True)
            
# Cao best spe
part_res_modes.groupby('init').mean() 
part_res_modes.groupby('init').std() 

part_res_modes.to_csv(res_folder + '/part_res_modes.csv')

#****************************
# K prototypes
#****************************

part_res_proto = pd.DataFrame(columns = ['it_id', 'init', 'micro', 'macro', 'purity'])


for init in inits:
    print(init)
    for i in range(nb_trials):
        km = KPrototypes(n_clusters = n_clusters, init = init, n_init=10, verbose=0)
        kmo_labels = km.fit_predict(y_np_nenc, categorical = np.where(cf_non_enc)[0].tolist())
        m, pred = misc(labels_oh, kmo_labels, True) 
        micro = precision_score(labels_oh, pred, average = 'micro')
        macro = precision_score(labels_oh, pred, average = 'macro')
        cm = confusion_matrix(labels_oh, pred)
        purity = cluster_purity(cm)

        part_res_proto = part_res_proto.append({'it_id': i + 1, 'init': init, \
                            'micro': micro, 'macro': macro, 'purity': purity}, \
                                               ignore_index=True)

# Random is best
part_res_proto.groupby('init').mean()
part_res_proto.groupby('init').std()

part_res_proto.to_csv(res_folder + '/part_res_proto.csv')

#****************************
# Hierarchical clustering
#****************************

hierarch_res = pd.DataFrame(columns = ['it_id', 'linkage', 'micro', 'macro', 'purity'])

linkages = ['complete', 'average', 'single']

for linky in linkages: 
    for i in range(nb_trials):  
        aglo = AgglomerativeClustering(n_clusters = n_clusters, affinity ='precomputed', linkage = linky)
        aglo_preds = aglo.fit_predict(dm)
        m, pred = misc(labels_oh, aglo_preds, True) 
        micro = precision_score(labels_oh, pred, average = 'micro')
        macro = precision_score(labels_oh, pred, average = 'macro')
        cm = confusion_matrix(labels_oh, pred)
        purity = cluster_purity(cm)

        hierarch_res = hierarch_res.append({'it_id': i + 1, 'linkage': linky, \
                            'micro': micro, 'macro': macro, 'purity': purity},\
                                           ignore_index=True)

 
hierarch_res.groupby('linkage').mean()
hierarch_res.groupby('linkage').std()

hierarch_res.to_csv(res_folder + '/hierarch_res.csv')

#****************************
# Neural-network based
#****************************

som_res = pd.DataFrame(columns = ['it_id', 'sigma', 'lr' ,'micro', 'macro', 'purity'])
y_np = y.values
numobs = len(y)

sigmas = np.linspace(0.001, 3, 5)
lrs = np.linspace(0.0001, 0.5, 10)

for sig in sigmas:
    for lr in lrs:
        for i in range(nb_trials):
            som = MiniSom(n_clusters, 1, y_np.shape[1], sigma = sig, learning_rate = lr) # initialization of 6x6 SOM
            som.train(y_np, 100) # trains the SOM with 100 iterations
            som_labels = [som.winner(y_np[i])[0] for i in range(numobs)]
            m, pred = misc(labels_oh, som_labels, True) 
            cm = confusion_matrix(labels_oh, pred)
            micro = precision_score(labels_oh, pred, average = 'micro')
            macro = precision_score(labels_oh, pred, average = 'macro')
            purity = cluster_purity(cm)

            som_res = som_res.append({'it_id': i + 1, 'sigma': sig, 'lr': lr, \
                            'micro': micro, 'macro': macro, 'purity': purity},\
                                     ignore_index=True)
   
som_res.groupby(['sigma', 'lr']).mean()
som_res.groupby(['sigma', 'lr']).std()
som_res.to_csv(res_folder + '/som_res.csv')


#****************************
# Other algorithms family
#****************************

ss = StandardScaler()
y_scale = ss.fit_transform(y_np)

dbs_res = pd.DataFrame(columns = ['it_id', 'data' ,'leaf_size', 'eps', 'min_samples','micro', 'macro', 'purity'])

lf_size = np.arange(1,6) * 10
epss = np.linspace(0.01, 5, 5)
min_ss = np.arange(1, 5)
data_to_fit = ['scaled', 'gower']

for lfs in lf_size:
    print("Leaf size:", lfs)
    for eps in epss:
        for min_s in min_ss:
            for data in data_to_fit:
                for i in range(1):
                    if data == 'gower':
                        dbs = DBSCAN(eps = eps, min_samples = min_s, metric = 'precomputed', leaf_size = lfs).fit(dm)
                    else:
                        dbs = DBSCAN(eps = eps, min_samples = min_s, leaf_size = lfs).fit(y_scale)
                        
                    dbs_preds = dbs.labels_
                    
                    if len(np.unique(dbs_preds)) > n_clusters:
                        continue
                    
                    m, pred = misc(labels_oh, dbs_preds, True) 
                    cm = confusion_matrix(labels_oh, pred)
                    micro = precision_score(labels_oh, pred, average = 'micro')
                    macro = precision_score(labels_oh, pred, average = 'macro')
                    purity = cluster_purity(cm)
    
                    dbs_res = dbs_res.append({'it_id': i + 1, 'leaf_size': lfs, \
                                'eps': eps, 'min_samples': min_s, 'micro': micro,\
                                    'data': data, 'macro': macro, 'purity': purity},\
                                             ignore_index=True)

# scaled data eps = 3.7525 and min_samples = 4  is the best spe
mean_res = dbs_res.groupby(['data','leaf_size', 'eps', 'min_samples']).mean()
dbs_res[(dbs_res['eps'] == 3.7525) & (dbs_res['data'] == 'scaled')].groupby(['leaf_size']).mean()

dbs_res[(dbs_res['eps'] == 3.7525) & (dbs_res['data'] == 'scaled')].groupby(['leaf_size']).std()

dbs_res.to_csv(res_folder + '/dbs_res.csv')