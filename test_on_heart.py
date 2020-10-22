# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 13:25:11 2020

@author: rfuchs
"""

import os 

os.chdir('C:/Users/rfuchs/Documents/GitHub/MDGMM')

from copy import deepcopy
from gower import gower_matrix

from sklearn.metrics import silhouette_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder


import pandas as pd

from mdgmm import MDGMM
from utilities import check_inputs
from init_params import dim_reduce_init
from metrics import misc
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
os.chdir('C:/Users/rfuchs/Documents/GitHub/MDGMM/datasets')

heart = pd.read_csv('heart_statlog/heart.csv', sep = ' ', header = None)
y = heart.iloc[:,:-1]
labels = heart.iloc[:,-1]
labels = np.where(labels == 1, 0, labels)
labels = np.where(labels == 2, 1, labels)

y = y.infer_objects()
numobs = len(y)

# Too many zeros for this "continuous variable". Add a little noise to avoid 
# the correlation matrix for each group to blow up
uniform_draws = uniform(0, 1E-12, numobs)
y.iloc[:, 9] = np.where(y[9] == 0, uniform_draws, y[9])

p = y.shape[1]

#===========================================#
# Formating the data
#===========================================#
var_distrib = np.array(['continuous', 'bernoulli', 'categorical', 'continuous',\
                        'continuous', 'bernoulli', 'categorical', 'continuous',\
                        'bernoulli', 'continuous', 'ordinal', 'ordinal',\
                        'categorical']) 
    
# Ordinal data already encoded

y_categ_non_enc = deepcopy(y)
vd_categ_non_enc = deepcopy(var_distrib)

# Encode categorical datas
y, var_distrib = gen_categ_as_bin_dataset(y, var_distrib)

# Encode binary data
le = LabelEncoder()
for col_idx, colname in enumerate(y.columns):
    if var_distrib[col_idx] == 'bernoulli': 
        y[colname] = le.fit_transform(y[colname])
    
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
n_clusters = 2
r = {'c': [nb_cont], 'd': [3], 't': [2, 1]}
k = {'c': [1], 'd': [1], 't': [n_clusters, 1]}

seed = 1
init_seed = 2
    
eps = 1E-05
it = 15
maxstep = 100

#n_clusters = len(np.unique(labels))

dtype = {y.columns[j]: np.float64 if (var_distrib[j] != 'bernoulli') & \
        (var_distrib[j] != 'categorical') else np.str for j in range(p_new)}

y = y.astype(dtype, copy=True)

# MCA init
prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None)
m, pred = misc(labels_oh, prince_init['classes'], True) 
print(m)
print(confusion_matrix(labels_oh, pred))
print('Silhouette', silhouette_score(dm, pred, metric = 'precomputed'))


'''
init = prince_init
y = y_np
perform_selec = True
'''

'''
import warnings
warnings.simplefilter('ignore')
'''

out = MDGMM(y_np, n_clusters, r, k, prince_init, var_distrib, nj, it, eps,\
            maxstep, seed, perform_selec = True)
m, pred = misc(labels_oh, out['classes'], True) 
print(m)
print(confusion_matrix(labels_oh, pred))
print('Silhouette', silhouette_score(dm, pred, metric = 'precomputed'))


#===========================================#
# Final plotting
#===========================================# 

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


#===========================================#
# Try auto mode
#===========================================# 
r = {'c': [nb_cont], 'd': [3], 't': [2, 1]}
k = {'c': [1], 'd': [2], 't': [4, 1]}

prince_init = dim_reduce_init(y, 'auto', k, r, nj, var_distrib, seed = None)
out = MDGMM(y_np, 'auto', r, k, prince_init, var_distrib, nj, it, eps,\
            maxstep, seed, perform_selec = True)
print('Silhouette', silhouette_score(dm, out['classes'], metric = 'precomputed'))

#===========================================#
# Try multi mode
#===========================================# 
n_clusters = 'multi'

r = {'c': [nb_cont], 'd': [5, 4], 't': [3, 2, 1]}
k = {'c': [1], 'd': [2, 2], 't': [3, 2, 1]}

prince_init = dim_reduce_init(y, 'multi', k, r, nj, var_distrib, seed = None)
out = MDGMM(y_np, 'multi', r, k, prince_init, var_distrib, nj, it, eps,\
            maxstep, seed, perform_selec = False)

# Check that all labels exist at each layer
Lt = len(k['t']) - 1

for l in range(Lt):
    nb_classes_found = len(np.unique(out['classes'][l]))
    print('Layer', l, 'was looking for', k['t'][l], 'groups,', \
          nb_classes_found, 'found')
    if nb_classes_found >= 2:
        sil = silhouette_score(dm, out['classes'][l], metric = 'precomputed')
        print('Silhouette coefficient is groups is', sil)
        

#=============================================#
# As a Feature extractor
#==============================================#
from prince import FAMD
from lightgbm import LGBMClassifier

#**********************************
# Extract deep features from MDGMM
#**********************************

r = {'c': [nb_cont], 'd': [3], 't': [2, 1]}
k = {'c': [1], 'd': [2], 't': [2, 1]}

prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None)
out = MDGMM(y_np, n_clusters, r, k, prince_init, var_distrib, nj, it, eps,\
            maxstep, seed, perform_selec = True)

mdgmm_dp = out['z']

#**********************************
# Extract deep features from FAMD
#**********************************

famd = FAMD(n_components = 2,\
    n_iter=3, copy=True,\
    check_input=True, engine='auto')
famd_dp = famd.fit_transform(y).values 

#**********************************
# Fit the LGBM
#**********************************

lgbm = LGBMClassifier(objective = 'binary')
mdgmm_fx = cross_validate(lgbm, mdgmm_dp, labels_oh.astype(int), cv = 5, scoring  = 'accuracy')
print('MDGMM test score', np.mean(mdgmm_fx['test_score']))

lgbm = LGBMClassifier(objective = 'binary')
famd_fx = cross_validate(lgbm, famd_dp, labels_oh.astype(int), cv = 5, scoring  = 'accuracy')
print('FAMD test score', np.mean(famd_fx['test_score']))


#=========================================================================
# Performance measure : Finding the best specification for init and MDGMM
#=========================================================================
res_folder = 'C:/Users/rfuchs/Documents/These/Experiences/mixed_algos/heart'


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
        try:
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
        except:
            mca_mdgmm_res = mca_mdgmm_res.append({'it_id': i + 1, 'r': str(r),\
                                                  'k': k,\
                                                  'micro': np.nan, 'macro': np.nan, \
                                                  'silhouette': np.nan},\
                                                   ignore_index=True)
            
           
mca_mdgmm_res.groupby('r').mean().max()
mca_mdgmm_res.groupby('r').std()

mca_mdgmm_res.to_csv(res_folder + '/mca_mdgmm_res.csv')

#============================================
# MDGMM. Thresholds use: ? and ?
# r {'d': [4], 't': [3, 1], 'c': [5]}
# k {'d': [2], 't': [2, 1], 'c': [1]}
#============================================
res_folder = 'C:/Users/rfuchs/Documents/These/Experiences/mixed_algos/heart'

# First find the best architecture 
numobs = len(y)
r = {'c': [nb_cont], 'd': [5], 't': [4, 3]}
k = {'c': [1], 'd': [1], 't': [n_clusters, 1]}

eps = 1E-05
it = 30
maxstep = 100

prince_init = dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None)
out = MDGMM(y_np, n_clusters, r, k, prince_init, var_distrib, nj, it, eps,\
            maxstep, seed = None)

r = out['best_r']
numobs = len(y)
k = out['best_k']
eps = 1E-05
it = 30
maxstep = 300

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
mdgmm_res.mean().max()

mdgmm_res.std()

mdgmm_res.to_csv(res_folder + '/mdgmm_res_kd1_autoselec.csv')

mdgmm_res = pd.read_csv(res_folder + '/mdgmm_res_kd1_autoselec.csv')

#=======================================================================
# Performance measure : Finding the best specification for other algos
#=======================================================================

from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import AgglomerativeClustering
from minisom import MiniSom   
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# <nb_trials> tries for each specification
nb_trials = 30

res_folder = 'C:/Users/rfuchs/Documents/These/Experiences/mixed_algos/heart'

#****************************
# Partitional algorithm
#****************************

part_res_modes = pd.DataFrame(columns = ['it_id', 'init', 'micro', 'macro', 'silhouette'])

inits = ['Huang', 'Cao', 'random']

for init in inits:
    print(init)
    for i in range(nb_trials):
        km = KModes(n_clusters= n_clusters, init=init, n_init=10, verbose=0)
        kmo_labels = km.fit_predict(y_np_nenc)
        m, pred = misc(labels_oh, kmo_labels, True)
        
        sil = silhouette_score(dm, pred, metric = 'precomputed')            
        micro = precision_score(labels_oh, pred, average = 'micro')
        macro = precision_score(labels_oh, pred, average = 'macro')

        part_res_modes = part_res_modes.append({'it_id': i + 1, 'init': init, \
                            'micro': micro, 'macro': macro, 'silhouette': sil}, \
                                               ignore_index=True)
            
# Cao best spe
part_res_modes.groupby('init').mean() 
part_res_modes.groupby('init').std() 

part_res_modes.to_csv(res_folder + '/part_res_modes.csv')

#****************************
# K prototypes
#****************************

part_res_proto = pd.DataFrame(columns = ['it_id', 'init', 'micro', 'macro', 'silhouette'])


for init in inits:
    print(init)
    for i in range(nb_trials):
        km = KPrototypes(n_clusters = n_clusters, init = init, n_init=10, verbose=0)
        kmo_labels = km.fit_predict(y_np_nenc, categorical = np.where(cf_non_enc)[0].tolist())
        m, pred = misc(labels_oh, kmo_labels, True) 
        
        sil = silhouette_score(dm, pred, metric = 'precomputed')            
        micro = precision_score(labels_oh, pred, average = 'micro')
        macro = precision_score(labels_oh, pred, average = 'macro')

        part_res_proto = part_res_proto.append({'it_id': i + 1, 'init': init, \
                            'micro': micro, 'macro': macro, 'silhouette': sil}, \
                                               ignore_index=True)

# Random is best
part_res_proto.groupby('init').mean()
part_res_proto.groupby('init').std()

part_res_proto.to_csv(res_folder + '/part_res_proto.csv')

#****************************
# Hierarchical clustering
#****************************

hierarch_res = pd.DataFrame(columns = ['it_id', 'linkage', 'micro', 'macro', 'silhouette'])

linkages = ['complete', 'average', 'single']

for linky in linkages: 
    for i in range(nb_trials):  
        aglo = AgglomerativeClustering(n_clusters = n_clusters, affinity ='precomputed', linkage = linky)
        aglo_preds = aglo.fit_predict(dm)
        m, pred = misc(labels_oh, aglo_preds, True) 

        sil = silhouette_score(dm, pred, metric = 'precomputed')            
        micro = precision_score(labels_oh, pred, average = 'micro')
        macro = precision_score(labels_oh, pred, average = 'macro')


        hierarch_res = hierarch_res.append({'it_id': i + 1, 'linkage': linky, \
                            'micro': micro, 'macro': macro, 'silhouette': sil},\
                                           ignore_index=True)

 
hierarch_res.groupby('linkage').mean()
hierarch_res.groupby('linkage').std()

hierarch_res.to_csv(res_folder + '/hierarch_res.csv')

#****************************
# Neural-network based
#****************************

som_res = pd.DataFrame(columns = ['it_id', 'sigma', 'lr' ,'micro', 'macro', 'silhouette'])
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
            
            try: # If only one class sil is not defined
                sil = silhouette_score(dm, pred, metric = 'precomputed')
            except ValueError:
                sil = np.nan
                
            micro = precision_score(labels_oh, pred, average = 'micro')
            macro = precision_score(labels_oh, pred, average = 'macro')

            som_res = som_res.append({'it_id': i + 1, 'sigma': sig, 'lr': lr, \
                            'micro': micro, 'macro': macro, 'silhouette': sil},\
                                     ignore_index=True)
   
som_res.groupby(['sigma', 'lr']).mean().max()
som_res.groupby(['sigma', 'lr']).std()
som_res.to_csv(res_folder + '/som_res.csv')


#****************************
# Other algorithms family
#****************************

ss = StandardScaler()
y_scale = ss.fit_transform(y_np)

dbs_res = pd.DataFrame(columns = ['it_id', 'data' ,'leaf_size', 'eps',\
                                  'min_samples','micro', 'macro', 'silhouette'])

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
                        dbs = DBSCAN(eps = eps, min_samples = min_s, \
                                     metric = 'precomputed', leaf_size = lfs).fit(dm)
                    else:
                        dbs = DBSCAN(eps = eps, min_samples = min_s, leaf_size = lfs).fit(y_scale)
                        
                    dbs_preds = dbs.labels_
                    
                    if len(np.unique(dbs_preds)) > n_clusters:
                        continue
                    
                    m, pred = misc(labels_oh, dbs_preds, True) 
                    
                    try: # If only one class sil is not defined
                        sil = silhouette_score(dm, pred, metric = 'precomputed')
                    except ValueError:
                        sil = np.nan
                        
                    micro = precision_score(labels_oh, pred, average = 'micro')
                    macro = precision_score(labels_oh, pred, average = 'macro')

                    dbs_res = dbs_res.append({'it_id': i + 1, 'leaf_size': lfs, \
                                'eps': eps, 'min_samples': min_s, 'micro': micro,\
                                    'data': data, 'macro': macro, 'silhouette': sil},\
                                             ignore_index=True)

# scaled data eps = 3.7525 and min_samples = 4  is the best spe
mean_res = dbs_res.groupby(['data','leaf_size', 'eps', 'min_samples']).mean()
maxs = mean_res.max()

mean_res[mean_res['micro'] == maxs['micro']].std()
mean_res[mean_res['silhouette'] == maxs['silhouette']].std()

dbs_res.to_csv(res_folder + '/dbs_res.csv')
