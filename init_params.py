# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 16:55:44 2020

@author: RobF
"""

import os
os.chdir('C:/Users/rfuchs/Documents/GitHub/MDGMM')

from copy import deepcopy
from itertools import product

from identifiability_DGMM import identifiable_estim_DDGMM, compute_z_moments,\
        diagonal_cond
        
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from factor_analyzer import FactorAnalyzer
from sklearn.mixture import GaussianMixture

from sklearn.preprocessing import StandardScaler 

from data_preprocessing import bin_to_bern
from utilities import compute_path_params, isnumeric
    
import prince
import pandas as pd
from sklearn.cross_decomposition import PLSRegression


# Dirty local hard copy of the Github bevel package
from bevel.linear_ordinal_regression import  OrderedLogit 

import autograd.numpy as np
from autograd.numpy import newaxis as n_axis
from autograd.numpy.linalg import LinAlgError

import warnings 
warnings.simplefilter('default')

####################################################################################
################### MCA GMM + Logistic Regressions initialisation ##################
####################################################################################

def add_missing_paths(k, init_paths, init_nb_paths):
    ''' Add the paths that have been given zeros probabily during init '''
    
    L = len(k)
    all_possible_paths = list(product(*[np.arange(k[l]) for l in range(L)]))
    existing_paths = [tuple(path.astype(int)) for path in init_paths] # Turn them as a list of tuple
    nb_existing_paths = deepcopy(init_nb_paths)
        
    for idx, path in enumerate(all_possible_paths):
        if not(path in existing_paths):
            #print('The following path has been added', idx, path)
            existing_paths.insert(idx, path)
            nb_existing_paths = np.insert(nb_existing_paths, idx, 0, axis = 0)

    return existing_paths, nb_existing_paths

'''
,, 

zl = zt[l]
kl = k['t'][l]
rl_nextl = r['t'][l:]
'''

def get_MFA_params(zl, kl, rl_nextl):
    ''' Determine clusters with a GMM and then adjust a Factor Model over each cluster
    zl (ndarray): The lth layer latent variable 
    kl (int): The number of components of the lth layer
    rl_nextl (1darray): The dimension of the lth layer and (l+1)th layer
    -----------------------------------------------------
    returns (dict): Dict with the parameters of the MFA approximated by GMM + FA. 
    '''

    numobs = zl.shape[0]
    not_all_groups = True

    max_trials = 100
    empty_count_counter = 0

    #======================================================
    # Fit a GMM in the continuous space
    #======================================================
    
    while not_all_groups:
        # If not enough obs per group then the MFA diverge...    

        gmm = GaussianMixture(n_components = kl)
        s = gmm.fit_predict(zl)
        
        clusters_found, count = np.unique(s, return_counts = True)

        if (len(clusters_found) == kl) & (count > 10).all():
            not_all_groups = False
            
        empty_count_counter += 1
        if empty_count_counter >= max_trials:
            raise RuntimeError('Could not find a GMM init that presents the \
                               proper number of groups:', kl)
    
    psi = np.full((kl, rl_nextl[0], rl_nextl[0]), 0).astype(float)
    H = np.full((kl, rl_nextl[0], rl_nextl[1]), 0).astype(float)
    eta = np.full((kl, rl_nextl[0]), 0).astype(float)
    z_nextl = np.full((numobs, rl_nextl[1]), np.nan).astype(float)
  
    #========================================================
    # And then a FA on each of those group
    #========================================================

    for j in range(kl):
        indices = (s == j)
        fa = FactorAnalyzer(rotation = None, method = 'ml', n_factors = rl_nextl[1])
        try:
            fa.fit(zl[indices])
        except LinAlgError:
            #print(zl[indices])
            raise RuntimeError("Eigenvalues could not converge. It might be due \
                               to the fact that one of the continuous variable \
                            presented no withing-group variation. Check your \
                            continuous variables with value_counts for instance \
                            and look to repeated values")
                            
        psi[j] = np.diag(fa.get_uniquenesses())
        #psi[j] = fa.get_uniquenesses()

        H[j] = fa.loadings_
        z_nextl[indices] = fa.transform(zl[indices])
        eta[j] = np.mean(zl[indices], axis = 0)
                
    params = {'H': H, 'psi': psi, 'z_nextl': z_nextl, 'eta': eta, 'classes': s}
    return params


'''
zh_first = z1D
kh = k['d']
rh = r['d']
Lh = L['d']
'''

def init_head(zh_first, kh, rh, numobs, Lh):
    zh = [zh_first]
    eta = []
    H = []
    psi = []
    paths_pred = []

    for l in range(Lh - 1): 
        #print('l layer of init head is ', l)
        params = get_MFA_params(zh[l], kh[l], rh[l:])
            
        eta.append(params['eta'][..., n_axis])
        H.append(params['H'])
        psi.append(params['psi'])
        zh.append(params['z_nextl']) 
        paths_pred.append(params['classes'])
        
    return eta, H, psi, zh, paths_pred


def init_junction_layer(r, k, zc, zd):
    
    # Stack together all the latent variables from both heads  
    last_zcd = np.hstack([zc[-1], zd[-1]])
    
    # Create the latent variable of the junction layer by PCA
    pca = PCA(n_components= r['t'][0])
    zt_first = pca.fit_transform(last_zcd)
    
    eta = {}
    H = {}
    psi = {}
    paths_pred = {}
    
    for h in ['c', 'd']:
        last_kh = k[h][-1]
        last_zh = zc[-1] if h == 'c' else zd[-1]
        
        
        ###########################################
        not_all_groups = True

        max_trials = 100
        empty_count_counter = 0

        #======================================================
        # Fit a GMM in the continuous space
        #======================================================
        
        while not_all_groups:
            # If not enough obs per group then the MFA diverge...    
    
            gmm_h = GaussianMixture(n_components = last_kh)
            s_h = gmm_h.fit_predict(last_zh)
            paths_pred[h] = s_h
            
            clusters_found, count = np.unique(s_h, return_counts = True)
    
            if (len(clusters_found) == last_kh) & (count > 10).all():
                not_all_groups = False
                
            empty_count_counter += 1
            if empty_count_counter >= max_trials:
                raise RuntimeError('Could not find a GMM init that presents the \
                                   proper number of groups:', last_kh)
        
        ###############################################
        '''
        # Find groups among each head latent variable
        gmm_h = GaussianMixture(n_components = last_kh)
        s_h = gmm_h.fit_predict(last_zh)
        paths_pred[h] = s_h
        
        clusters_found, count = np.unique(s_h, return_counts = True)
        '''
        
        assert len(clusters_found) == last_kh

        eta_h = []
        H_h = []
        psi_h = []
        
        # Fill the parameters belonging to each group
    
        for j in range(last_kh):
            indices = (s_h == j)
            eta_hj = last_zh[indices].mean(axis = 0, keepdims = True)
            eta_h.append(eta_hj.T)
            centered_zh = last_zh[indices] - eta_hj
            
            # For each group fit a PLS to find Lambda and Psi
            # Choose r['t] - 1 but could take less 
            pls = PLSRegression(n_components = r['t'][0] - 1)
            pls.fit(zt_first[indices], centered_zh)

            
            H_hj = (pls.x_weights_ @ pls.y_loadings_.T).T
            H_h.append(H_hj)
            
            residuals_h = (pls.predict(zt_first[indices]) - centered_zh).T
            psi_h.append(np.cov(residuals_h))
           
        eta[h] = np.stack(eta_h)
        H[h] = np.stack(H_h)
        psi[h] = np.stack(psi_h)
        
    return eta, H, psi, paths_pred, zt_first
    
    


def dim_reduce_init(y, n_clusters, k, r, nj, var_distrib, seed = None):
    ''' Perform dimension reduction into a continuous r dimensional space and determine 
    the init coefficients in that space
    
    y (numobs x p ndarray): The observations containing categorical variables
    k (1d array): The number of components of the latent Gaussian mixture layers
    r (int): The dimension of latent variables
    nj (p 1darray): For binary/count data: The maximum values that the variable can take. 
                    For ordinal data: the number of different existing categories for each variable
    var_distrib (p 1darray): An array containing the types of the variables in y 
    dim_red_method (str): Choices are 'prince' for MCA, 'umap' of 'tsne'
    seed (None): The random state seed to use for the dimension reduction
    M (int): The number of MC points to compute     
    ---------------------------------------------------------------------------------------
    returns (dict): All initialisation parameters
    '''
    
    if type(y) != pd.core.frame.DataFrame:
        raise TypeError('y should be a dataframe for prince')

    numobs = len(y)
    
    # Length of both heads and tail
    L = {'c': len(k['c']), 'd': len(k['d']), 't': len(k['t']) - 1}

    # Paths of both heads and tail
    S = {'c': np.prod(k['c']), 'd': np.prod(k['d']), 't': np.prod(k['t'])}
            
    # Data of both heads 
    yc = y.iloc[:, var_distrib == 'continuous'].values
    yd = y.iloc[:, var_distrib != 'continuous'].values
    
    #==============================================================
    # Dimension reduction performed with MCA on discrete data
    #==============================================================

    # Check input = False to remove
    mca = prince.MCA(n_components = r['d'][0], n_iter=3, copy=True,\
                     check_input=False, engine='auto', random_state = seed)
    z1D = mca.fit_transform(yd.astype(str)).values
        
    # Be careful: The first z^c is the continuous data whether the first 
    # z^d is the MCA transformed data.
    #z = {'c': [yc], 'd': [z1D]}
    y = y.values   
    
    #==============================================================
    # Set the shape parameters of each discrete data type
    #==============================================================    
    
    y_bin = y[:, np.logical_or(var_distrib == 'bernoulli', var_distrib == 'binomial')]
    y_bin = y_bin.astype(int)
    nj_bin = nj[np.logical_or(var_distrib == 'bernoulli',var_distrib == 'binomial')]
    nb_bin = len(nj_bin)
    
    y_categ = y[:, var_distrib == 'categorical']
    nj_categ = nj[var_distrib == 'categorical']
    nb_categ = len(nj_categ)
    
    y_ord = y[:, var_distrib == 'ordinal']  
    y_ord = y_ord.astype(int)
    nj_ord = nj[var_distrib == 'ordinal']
    nb_ord = len(nj_ord)
    
    #yc = yc / np.std(yc.astype(np.float), axis = 0, keepdims = True)
    ss = StandardScaler()
    yc = ss.fit_transform(yc)
             
    #=======================================================
    # Determining the Gaussian Parameters
    #=======================================================
    init = {}

    # Initialise both heads quantities
    eta_d, H_d, psi_d, zd, paths_pred_d = init_head(z1D, k['d'], r['d'], numobs, L['d'])
    eta_c, H_c, psi_c, zc, paths_pred_c = init_head(yc, k['c'], r['c'], numobs, L['c'])
      
    # Initialisation of the common layer. The coefficients are those between the last
    # Layer of both heads and the first junction layer
    eta_h_last, H_h_last, psi_h_last, paths_pred_h_last, zt_first = init_junction_layer(r, k, zc, zd)
    eta_d.append(eta_h_last['d'])
    H_d.append(H_h_last['d'])
    psi_d.append(psi_h_last['d'])
    
    eta_c.append(eta_h_last['c'])
    H_c.append(H_h_last['c'])
    psi_c.append(psi_h_last['c'])
    
    paths_pred_d.append(paths_pred_h_last['d'])
    paths_pred_c.append(paths_pred_h_last['c'])
    zt = [zt_first]  
    
    # Initialisation of the following common layers 
    for l in range(L['t']):
        params = get_MFA_params(zt[l], k['t'][l], r['t'][l:]) 
        eta_c.append(params['eta'][..., n_axis])
        eta_d.append(params['eta'][..., n_axis])
    
        H_c.append(params['H'])
        H_d.append(params['H'])   
        
        psi_c.append(params['psi'])
        psi_d.append(params['psi'])  
        
        zt.append(params['z_nextl'])
        zc.append(params['z_nextl'])
        zd.append(params['z_nextl'])
        
        paths_pred_c.append(params['classes'])
        paths_pred_d.append(params['classes'])    
    
    paths_pred_c = np.stack(paths_pred_c).T
    paths_c, nb_paths_c = np.unique(paths_pred_c, return_counts = True, axis = 0)
    paths_c, nb_paths_c = add_missing_paths(k['c'] + k['t'][:-1], paths_c, nb_paths_c)
    
    paths_pred_d = np.stack(paths_pred_d).T
    paths_d, nb_paths_d = np.unique(paths_pred_d, return_counts = True, axis = 0)
    paths_d, nb_paths_d = add_missing_paths(k['d'] + k['t'][:-1], paths_d, nb_paths_d)
       
    w_s_c = nb_paths_c / numobs
    w_s_c = np.where(w_s_c == 0, 1E-16, w_s_c)
    
    w_s_d = nb_paths_d / numobs
    w_s_d = np.where(w_s_d == 0, 1E-16, w_s_d)
    
    # Check that all paths have been explored
    if (len(paths_c) != S['c'] * S['t']) | (len(paths_d) != S['d'] * S['t']):
        raise RuntimeError('Path initialisation failed')

    #=============================================================
    # Enforcing identifiability constraints over the first layer
    #=============================================================
    
    H_c = diagonal_cond(H_c, psi_c) # Hack to remove
    H_d = diagonal_cond(H_d, psi_d)

    # Recompute the mu and psi
    mu_s_c, sigma_s_c = compute_path_params(eta_c, H_c, psi_c)
    mu_s_d, sigma_s_d = compute_path_params(eta_d, H_d, psi_d)
    
    # Hack to remove    
    Ez1_c, AT_c = compute_z_moments(w_s_c, mu_s_c, sigma_s_c)        
    eta_c, H_c, psi_c = identifiable_estim_DDGMM(eta_c, H_c, psi_c, Ez1_c, AT_c)
    
    Ez1_d, AT_d = compute_z_moments(w_s_d, mu_s_d, sigma_s_d)
    eta_d, H_d, psi_d = identifiable_estim_DDGMM(eta_d, H_d, psi_d, Ez1_d, AT_d)

        
    init['c'] = {}
    init['c']['eta']  = eta_c     
    init['c']['H'] = H_c
    init['c']['psi'] = psi_c
    init['c']['w_s'] = w_s_c # Probabilities of each path through the network
    init['c']['z'] = zc
    
    init['d'] = {}
    init['d']['eta']  = eta_d     
    init['d']['H'] = H_d
    init['d']['psi'] = psi_d
    init['d']['w_s'] = w_s_d # Probabilities of each path through the network
    init['d']['z'] = zd
    
    
    # The clustering layer is the one used to perform the clustering 
    # i.e. the layer l such that k[l] == n_clusters
    if not(isnumeric(n_clusters)):
        if n_clusters == 'auto':
            #n_clusters = k['t'][0]
            # First tail layer is the default clustering layer in auto mode
            clustering_layer = L['c']
            
        elif n_clusters == 'multi':
            clustering_layer = range(L['t'])

        else:
            raise ValueError('Please enter an int, auto or multi for n_clusters')
    else:
        kc_complete = k['c'] + k['t'][:-1]
        common_clus_layer_idx = (np.array(kc_complete) == n_clusters)
        common_clus_layer_idx[:L['c']] = False    
        clustering_layer = np.argmax(common_clus_layer_idx)
    
        assert clustering_layer >= L['c']
    
    init['classes'] = paths_pred_c[:,clustering_layer] 

    
    
         
    #=======================================================
    # Determining the coefficients of the GLLVM layer
    #=======================================================
    
    # Determining lambda_bin coefficients.
    lambda_bin = np.zeros((nb_bin, r['d'][0] + 1))
    
    for j in range(nb_bin): 
        Nj = int(np.max(y_bin[:,j])) # The support of the jth binomial is [1, Nj]
        
        if Nj ==  1:  # If the variable is Bernoulli not binomial
            yj = y_bin[:,j]
            z_new = zd[0]
        else: # If not, need to convert Binomial output to Bernoulli output
            yj, z_new = bin_to_bern(Nj, y_bin[:,j], zd[0])
        
        lr = LogisticRegression()
        
        if j < r['d'][0] - 1:
            lr.fit(z_new[:,:j + 1], yj)
            lambda_bin[j, :j + 2] = np.concatenate([lr.intercept_, lr.coef_[0]])
        else:
            lr.fit(z_new, yj)
            lambda_bin[j] = np.concatenate([lr.intercept_, lr.coef_[0]])
    
    ## Identifiability of bin coefficients
    lambda_bin[:,1:] = lambda_bin[:,1:] @ AT_d[0] 
    
    # Determining lambda_ord coefficients
    lambda_ord = []
    
    for j in range(nb_ord):
        #Nj = len(np.unique(y_ord[:,j], axis = 0))  # The support of the jth ordinal is [1, Nj]
        yj = y_ord[:,j]
        
        ol = OrderedLogit()
        ol.fit(zd[0], yj)
        
        ## Identifiability of ordinal coefficients
        beta_j = (ol.beta_.reshape(1, r['d'][0]) @ AT_d[0]).flatten()
        lambda_ord_j = np.concatenate([ol.alpha_, beta_j])
        lambda_ord.append(lambda_ord_j)   
        
        
    # Determining lambda_categ coefficients
    lambda_categ = []
    
    for j in range(nb_categ):
        yj = y_categ[:,j]
        
        lr = LogisticRegression(multi_class = 'multinomial')
        lr.fit(zd[0], yj)        

        ## Identifiability of categ coefficients
        beta_j = lr.coef_ @ AT_d[0]  
        lambda_categ.append(np.hstack([lr.intercept_[...,n_axis], beta_j]))  
        
    init['lambda_bin'] = lambda_bin
    init['lambda_ord'] = lambda_ord
    init['lambda_categ'] = lambda_categ
    
    return init

