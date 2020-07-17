# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:39:05 2020

@author: rfuchs
"""

from data_preprocessing import bin_to_bern
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
# Dirty local hard copy of the Github bevel package
from bevel.linear_ordinal_regression import  OrderedLogit 
import warnings 


warnings.simplefilter('default')

import autograd.numpy as np

def rl1_selection(y_bin, y_ord, zl1_ys, w_s):

    M0 = zl1_ys.shape[0]
    numobs = zl1_ys.shape[1] 
    r0 = zl1_ys.shape[2]
    S0 = zl1_ys.shape[3] 

    nb_bin = y_bin.shape[1]
    nb_ord = y_ord.shape[1]
    
    # Define the hyperparameters: 
    ## PROP_ZERO_THRESHOLD : The limit proportion of time a coefficient has 
    ### been found to be zero before this dimension is deleted
    
    ## PVALUE_THRESHOLD: The p-value threshold to zero a coefficient in ordinal
    ### logistic regression 
        
    PROP_ZERO_THRESHOLD = 0.5
    PVALUE_THRESHOLD = 0.10
    
    # Detemine the dimensions that are weakest for Binomial variables
    zero_coef_mask = np.zeros(r0)
    for j in range(nb_bin):
        for s in range(S0):
            Nj = np.max(y_bin[:,j]) # The support of the jth binomial is [1, Nj]
            if Nj ==  1:  # If the variable is Bernoulli not binomial
                yj = y_bin[:,j]
                z = zl1_ys[:,:,:,s]
            else: # If not, need to convert Binomial output to Bernoulli output
                yj, z = bin_to_bern(Nj, y_bin[:,j], z[0])
        
            # Put all the M0 points in a series
            X = z.flatten(order = 'C').reshape((M0 * numobs, r0), order = 'C')
            y_repeat = np.repeat(yj, M0).astype(int) # Repeat rather than tile to check
            
            lr = LogisticRegression(penalty = 'l1', solver = 'saga')
            lr.fit(X, y_repeat)
            zero_coef_mask += (lr.coef_[0] == 0) * w_s[s]
    
    #print(zero_coef_mask)
    # Detemine the dimensions that are weakest for Ordinal variables
    for j in range(nb_ord):
        for s in range(S0):
            ol = OrderedLogit()
            X = zl1_ys[:,:,:,s].flatten(order = 'C').reshape((M0 * numobs, r0), order = 'C')
            y_repeat = np.repeat(y_ord[:, j], M0).astype(int) # Repeat rather than tile to check
            
            ol.fit(X, y_repeat)
            zero_coef_mask += np.array(ol.summary['p'] > PVALUE_THRESHOLD) * w_s[s]
    
    #print(zero_coef_mask)
    
    # Voting: Delete the dimensions which have been zeroed a majority of times 
    zeroed_coeff_prop = zero_coef_mask / ((nb_ord + nb_bin))
    
    # Need at least r1 = 2 for algorithm to work
    new_rl = np.sum(zeroed_coeff_prop <= PROP_ZERO_THRESHOLD)
    
    if new_rl < 2:
        dims_to_keep = np.argsort(zeroed_coeff_prop)[:2]
        
    else:
        dims_to_keep = list(set(range(r0))  - \
                        set(np.where(zeroed_coeff_prop > PROP_ZERO_THRESHOLD)[0].tolist()))

    return dims_to_keep


def other_r_selection(rl1_select, z2_z1s):

    S = [zz.shape[2] for zz in z2_z1s] + [1]    
    CORR_THRESHOLD = 0.20
    
    L = len(z2_z1s)
    M = np.array([zz.shape[0] for zz in z2_z1s] + [z2_z1s[-1].shape[1]])
    prev_new_r = [len(rl1_select)]
    
    dims_to_keep = []
    
    for l in range(L):        
        # Will not keep the following layers if one of the previous layer is of dim 1
        if prev_new_r[l] <= 1:
            dims_to_keep.append([])
            prev_new_r.append(0)
            
        else: 
            old_rl = z2_z1s[l].shape[-1]
            corr = np.zeros(old_rl)
            
            for s in range(S[l]):
                for m1 in range(M[l + 1]):
                    pca = PCA(n_components=1)
                    pca.fit_transform(z2_z1s[l][m1, :, s])
                    corr += np.abs(pca.components_[0])
            
            average_corr = corr / (S[l] * M[l + 1])
            new_rl = np.sum(average_corr > CORR_THRESHOLD)
        
            if prev_new_r[l] > new_rl: # Respect r1 > r2 > r3 ....
                dims_to_keep.append(np.where(average_corr > CORR_THRESHOLD)[0].tolist())
            else: # Have to delete other dimensions to match r1 > r2 > r3 ....
                nb_dims_to_remove = old_rl - prev_new_r[l] + 1
                unwanted_dims = np.argpartition(average_corr, nb_dims_to_remove)[:nb_dims_to_remove]
                wanted_dims = list(set(range(old_rl)) - set(unwanted_dims))
                dims_to_keep.append(wanted_dims)
                new_rl = len(wanted_dims)
                
            prev_new_r.append(new_rl)

    return dims_to_keep


def r_select(y_bin, y_ord, zl1_ys, z2_z1s, w_s):
    ''' Automatic choice of dimension of each layer components '''

    rl1_select = rl1_selection(y_bin, y_ord, zl1_ys, w_s)
    other_r_select =  other_r_selection(rl1_select, z2_z1s)
    return [rl1_select] + other_r_select

def k_select(w_s, k, new_L, clustering_layer):
    ''' Automatic choice of the number of components by layer '''
    
    L = len(k)
    n_clusters = k[clustering_layer]
    
    # If the clustering layer (cl) is deleted, define the last existing layer as cl 
    last_layer_idx = new_L - 1
    if last_layer_idx  < clustering_layer:
        clustering_layer = last_layer_idx
    
    components_to_keep = []
    w = w_s.reshape(*k, order = 'C')
            
    for l in range(new_L):
                
        PROBA_THRESHOLD = 1 / (k[l] * 4)

        other_layers_indices = tuple(set(range(L)) - set([l]))
        components_proba = w.sum(other_layers_indices)
        #print(components_proba)
        
        if l == clustering_layer:
            biggest_lik_comp = np.sort(components_proba.argsort()[::-1][:n_clusters])
            components_to_keep.append(biggest_lik_comp)

        else:
            components_to_keep.append(np.where(components_proba > PROBA_THRESHOLD)[0])
    
    return components_to_keep