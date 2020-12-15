# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:39:05 2020

@author: rfuchs
"""


from copy import deepcopy
from utilities import isnumeric
from sklearn.decomposition import PCA
from data_preprocessing import bin_to_bern
from sklearn.linear_model import LogisticRegression
from autograd.numpy import newaxis as n_axis

# Dirty local hard copy of the Github bevel package
from bevel.linear_ordinal_regression import  OrderedLogit 
import warnings 


warnings.simplefilter('default')

import autograd.numpy as np

def rl1_selection(y_bin, y_ord, y_categ, zl1_ys, w_s, Ld):
    ''' Selects the number of factors on the first latent discrete layer 
    y_bin (n x p_bin ndarray): The binary and count data matrix
    y_ord (n x p_ord ndarray): The ordinal data matrix
    y_categ (n x p_categ ndarray): The categorical data matrix
    zl1_ys (k_1D x r_1D ndarray): The first layer latent variables
    w_s (list): The path probabilities starting from the first discrete head layer
    Ld (list of int):  The number of layers on the discrete head   
    ------------------------------------------------------------------
    return (list of int): The dimensions to keep 
    '''

    M0 = zl1_ys.shape[0]
    numobs = zl1_ys.shape[1] 
    r0 = zl1_ys.shape[2]
    S0 = zl1_ys.shape[3] 

    nb_bin = y_bin.shape[1]
    nb_ord = y_ord.shape[1]
    nb_categ = y_categ.shape[1]
    
    # Need at least r1 == Ld for algorithm to work (to have r1 < r2 <r3 <...)
    min_arch_rl1 = Ld
    if r0 == min_arch_rl1: # TO CHECK
        return list(range(r0))
            
    PROP_ZERO_THRESHOLD = 0.25
    PVALUE_THRESHOLD = 0.10
    
    # Detemine the dimensions that are weakest for Binomial variables
    zero_coef_mask = np.zeros(r0)
    for j in range(nb_bin):
        for s in range(S0):
            Nj = int(np.max(y_bin[:,j])) # The support of the jth binomial is [1, Nj]
            
            if Nj ==  1:  # If the variable is Bernoulli not binomial
                yj = y_bin[:,j]
                z = zl1_ys[:,:,:,s]
            else: # If not, need to convert Binomial output to Bernoulli output
                yj, z = bin_to_bern(Nj, y_bin[:,j], zl1_ys[:,:,:,s])
                #yj, z = bin_to_bern(Nj, y_bin[:,j], z[0])
        
            # Put all the M0 points in a series
            X = z.flatten(order = 'C').reshape((-1, r0), order = 'C')

            y_repeat = np.repeat(yj, M0).astype(int) # Repeat rather than tile to check
            
            lr = LogisticRegression(penalty = 'l1', solver = 'saga')
            lr.fit(X, y_repeat)
            zero_coef_mask += (lr.coef_[0] == 0) * w_s[s]
    
    # Detemine the dimensions that are weakest for Ordinal variables
    for j in range(nb_ord):
        for s in range(S0):
            ol = OrderedLogit()
            X = zl1_ys[:,:,:,s].flatten(order = 'C').reshape((M0 * numobs, r0), order = 'C')
            y_repeat = np.repeat(y_ord[:, j], M0).astype(int) # Repeat rather than tile to check
            
            ol.fit(X, y_repeat)
            zero_coef_mask += np.array(ol.summary['p'] > PVALUE_THRESHOLD) * w_s[s]
                # Detemine the dimensions that are weakest for Categorical variables
                
    for j in range(nb_categ):
        for s in range(S0):
            z = zl1_ys[:,:,:,s]
                        
            # Put all the M0 points in a series
            X = z.flatten(order = 'C').reshape((M0 * numobs, r0), order = 'C')
            y_repeat = np.repeat(y_categ[:,j], M0).astype(int) # Repeat rather than tile to check
            
            lr = LogisticRegression(penalty = 'l1', solver = 'saga', \
                                    multi_class = 'multinomial')            
            lr.fit(X, y_repeat)  
            
            zero_coef_mask += (lr.coef_[0] == 0) * w_s[s]    
    
        
    # Voting: Delete the dimensions which have been zeroed a majority of times 
    zeroed_coeff_prop = zero_coef_mask / (nb_ord + nb_bin + nb_categ)
    new_rl = np.sum(zeroed_coeff_prop <= PROP_ZERO_THRESHOLD)
    
    # Keep enough coefficients for the model to be identifiable
    if new_rl < min_arch_rl1:
        dims_to_keep = np.argsort(zeroed_coeff_prop)[:min_arch_rl1]
        
    else:
        dims_to_keep = list(set(range(r0))  - \
                        set(np.where(zeroed_coeff_prop > PROP_ZERO_THRESHOLD)[0].tolist()))
    
    dims_to_keep = np.sort(dims_to_keep)
    
    return dims_to_keep


def other_r_selection(rl1_select, z2_z1s, Lt, head = True,\
                      mode_multi = False):
    ''' Chose the meaningful dimensions from the second layer of each head h/tail
    rl1_select (list): The dimension kept over the first layer of head/tail
    z2_z1s (list of ndarrays): z^{(l + 1)}| z^{(l)}, s
    Lt (list of int):  The number of layers on the common tail   
    head (Bool): Whether to determine head (True) or tail layers (False) dimensions
    mode_multi (Bool): Whether the algorithm is in multi_clus mode
    --------------------------------------------------------------------------
    return (list of int): The dimensions to keep from the second layer of the head/tail
    '''
    
    S = [zz.shape[2] for zz in z2_z1s] + [1] 
    CORR_THRESHOLD = 0.20
    
    Lh = len(z2_z1s)
    rh = [z2_z1s[l].shape[-1] for l in range(Lh)] 
    M = np.array([zz.shape[0] for zz in z2_z1s] + [z2_z1s[-1].shape[1]])
    prev_new_r = [len(rl1_select)]
    
    dims_to_keep = []
    dims_corr = [] # The correlations associated with the different dimensions
    
    for l in range(Lh):        
        # Will not keep the following layers if one of the previous layer is of dim 1
        if prev_new_r[l] <= 1:
            dims_to_keep.append([])
            prev_new_r.append(0)
            
        else: 
            old_rl = rh[l]
            corr = np.zeros(old_rl)
            
            for s in range(S[l]):
                for m1 in range(M[l + 1]):
                    pca = PCA(n_components=1)
                    pca.fit_transform(z2_z1s[l][m1, :, s])
                    corr += np.abs(pca.components_[0])
            
            average_corr = corr / (S[l] * M[l + 1])
            dims_corr.append(average_corr)
            
            new_rl = np.sum(average_corr > CORR_THRESHOLD)
            
            if new_rl < prev_new_r[l]: # Respect r1 > r2 > r3 ....
                # If multimode keep the same number of components and layer on the tail
                if mode_multi:
                    if head:
                        min_rl_for_viable_arch = Lh + Lt - (l + 1)
                    else:
                        min_rl_for_viable_arch = np.max(Lt - (l + 1), 0)
                else:
                    if head:
                        # If last layer of an head
                        if (Lh >= 1) & (l == Lh - 2): 
                            # If this layer is a bottleneck, we have to delete it
                            if new_rl <= 2: 
                                # Empty last head layer 
                                dims_to_keep.append([]) 
                                prev_new_r.append(0)
                                dims_corr[-1] = np.full(rh[l], 0.0)

                                # Tail layers remain the unchanged
                                for l1 in range(l + 1, Lh):
                                    dims_to_keep.append(list(range(rh[l1]))) 
                                    prev_new_r.append(rh[l1])
                                    dims_corr.append(np.full(rh[l1], 1.0))
                                break
                            else:
                                min_rl_for_viable_arch = new_rl
                            
                        else: # To adapt
                            min_rl_for_viable_arch = 2 + Lh - (l + 1)
                    else: 
                        min_rl_for_viable_arch = np.max(1 - l, 0)
                                        
                # Need to have an identifiable model but also a viable architecture
                if new_rl >= min_rl_for_viable_arch:
                    wanted_dims = np.where(average_corr > CORR_THRESHOLD)[0].tolist()
                else:
                    wanted_dims = np.argsort(- average_corr)[:min_rl_for_viable_arch]# -avg_score: In descending order 
                
                wanted_dims = np.sort(wanted_dims)
                dims_to_keep.append(deepcopy(wanted_dims))
                
            else: # Have to delete other dimensions to match r1 > r2 > r3 ....
                nb_dims_to_remove = old_rl - prev_new_r[l] + 1
                unwanted_dims = np.argpartition(average_corr, nb_dims_to_remove)[:nb_dims_to_remove]
                wanted_dims = list(set(range(old_rl)) - set(unwanted_dims))
                wanted_dims = np.sort(wanted_dims)
                dims_to_keep.append(deepcopy(wanted_dims))
                new_rl = len(wanted_dims)
                
            prev_new_r.append(new_rl)

    return dims_to_keep, dims_corr


def tail_r_selection(last_r_select_d, last_r_select_c, score_d, score_c):
    ''' Select the last dimensions of the tail layers
    last_r_select_d (list): The dimension selected on the last discrete layer
    last_r_select_c (list): The dimension selected on the last continuous layer
    score_d (list): The correlation between the first PCA axis of the first common 
                    tail layer and the last discrete layer 
    score_d (list): The correlation between the first PCA axis of the first common 
                    tail layer and the last continuous layer 
    --------------------------------------------------------------------------
    returns (list): The dimension kept of the first common tail layer
    '''
    
    avg_score = (score_c + score_d) / 2
    nb_dims_maxs = min(len(last_r_select_c), len(last_r_select_d)) # Keep it identifiable
    
    dims_kept = np.argsort(- avg_score)[:nb_dims_maxs] # -avg_score: In descending order        
    dims_kept = np.sort(dims_kept)

    return dims_kept


def r_select(y_bin, y_ord, y_categ, yc, zl1_ys_d, z2_z1s_d, w_s_d,\
             z2_z1s_c, z2_z1s_t, n_clusters):
    ''' Automatic choice of dimension of all layer components 
    y_bin (numobs x nb_bin nd-array): The binary/count data
    y_ord (numobs x nb_ord nd-array): The ordinal data
    y_categ (numobs x nb_categ nd-array): The categorical data
    yc (numobs x nb_categ nd-array): The continuous data
    zl1_ys_d (ndarray): The latent variable of the first discrete layer
    z2_z1s_* (list of ndarray): z^{l+1} | z^{l}, s, Theta for all (l,s) 
                                of the * head    
    w_s_d (list): The path probabilities for all s in [1,S^D]
    n_clusters (int): The number of clusters to look for in the data
    --------------------------------------------------------------------------
    returns (dict): The dimensions kept for all the layers of the network
    '''

    mode_multi = False

    if not(isnumeric(n_clusters)):
        if n_clusters == 'multi':
            mode_multi = True    
    
    Ld = len(z2_z1s_d) + len(z2_z1s_t) + 1 # +1 for the dim of the last layer of the tail
    Lt = len(z2_z1s_t) + 1
    
    r_selected = {}
    
    # Discrete head selection
    rl1_select_d = rl1_selection(y_bin, y_ord, y_categ, zl1_ys_d, w_s_d, Ld)
    other_r_select_d, dims_score_d =  other_r_selection(rl1_select_d, z2_z1s_d,\
                                                         Lt, mode_multi = mode_multi)
    r_selected['d'] = [rl1_select_d] + other_r_select_d[:-1]
    
    # Continuous head selection
    p = yc.shape[1]
    rl1_select_c = list(range(p))
    other_r_select_c, dims_score_c =  other_r_selection(rl1_select_c, z2_z1s_c,\
                                                        Lt, mode_multi = mode_multi)
    r_selected['c'] = [rl1_select_c] + other_r_select_c[:-1]
    
    # Common tail selection
    rl1_select_t =  tail_r_selection(other_r_select_d[-1], other_r_select_c[-1],\
                     dims_score_d[-1], dims_score_c[-1])
    
    # Min viable arch should not apply there ???
    other_r_select_t, dims_score_t =  other_r_selection(rl1_select_t, z2_z1s_t,\
                                                        Lt, head = False,\
                                                        mode_multi = mode_multi) 
    r_selected['t'] = [rl1_select_t] + other_r_select_t
    
    return r_selected



    
def k_select(w_s_c, w_s_d, w_s_t, k, new_Lt, clustering_layer, n_clusters):
    ''' Automatic choice of the number of components by layer 
    w_s_* (list): The path probabilities for all s in [1,S^*]
    k (dict): The number of component on each layer
    new_Lt (int): The selected number of layers on the common tail.
    clustering_layer (int): The index of the clustering layer
    n_clusters (int): The number of clusters to look for in the data
    --------------------------------------------------------------------------
    returns (dict): The components kept for all the layers of the network
    '''
    
    mode_auto = False
    mode_multi = False

    if not(isnumeric(n_clusters)):
        if n_clusters == 'auto':
            mode_auto = True
        elif n_clusters == 'multi':
            mode_multi = True
            
    w_s = {'c': w_s_c, 'd': w_s_d}
    Lt = len(k['t'])
    
    components_to_keep = {}
    
    #==============================================
    # Selection for both heads
    #==============================================
    
    for h in ['c', 'd']:
        Lh = len(k[h])
        w = w_s[h].reshape(*(k[h] + k['t']), order = 'C')  
        components_to_keep[h] = []
        
        for l in range(Lh):
            PROBA_THRESHOLD = 1 / (k[h][l] * 4)
    
            other_layers_indices = tuple(set(range(Lh + Lt)) - set([l]))
            components_proba = w.sum(other_layers_indices)
            comp_kept = np.where(components_proba > PROBA_THRESHOLD)[0]
            comp_kept = np.sort(comp_kept)
            
            components_to_keep[h].append(deepcopy(comp_kept))
    
    #==============================================
    # Selection for the tail
    #==============================================

    # Nb of components have to remain unchanged for multiclus mode
    if mode_multi: 
        components_to_keep['t'] = [list(range(k['t'][l])) for l in range(new_Lt)]
        return components_to_keep
    
    # If the clustering layer (cl) is deleted, define the last existing layer as cl 
    last_layer_idx = new_Lt - 2
    if last_layer_idx  < clustering_layer:
        clustering_layer = last_layer_idx
    
    w = w_s_t.reshape(*k['t'], order = 'C')
    components_to_keep['t'] = []
    
    for l in range(new_Lt):
                
        PROBA_THRESHOLD = 1 / (k['t'][l] * 4)

        other_layers_indices = tuple(set(range(Lt)) - set([l]))
        components_proba = w.sum(other_layers_indices)

        
        if (l == clustering_layer) & (not(mode_auto)): # Pb avec mode multi
            biggest_lik_comp = np.sort(components_proba.argsort()[::-1][:n_clusters])
            components_to_keep['t'].append(biggest_lik_comp)
        
        # If l is then end of the tail then it has to have only one component
        elif l == (new_Lt - 1):
            biggest_lik_comp = np.sort(components_proba.argsort()[::-1][:1])
            components_to_keep['t'].append(biggest_lik_comp)            

        else:
            comp_kept = np.where(components_proba > PROBA_THRESHOLD)[0]
            comp_kept = np.sort(comp_kept)
            components_to_keep['t'].append(comp_kept)
    
    return components_to_keep


def check_if_selection(r_to_keep, r, k_to_keep, k, L, new_Lt):
    ''' Check if the architecture has changed during the selection procedure
    r_to_keep (dict): The dimensions selected in the network
    r (dict): The original dimensions of the network layers
    k_to_keep (dict): The components selected in the network
    k (dict): The original number of component on each layer
    L (dict): The number of layers in the networks 
    new_Lt (int): The selected number of layers on the common tail.
    --------------------------------------------------------------------------
    returns (Bool): Whether or not the procedure has selected a new architecture
    '''
    
    is_L_unchanged = (L['t'] == new_Lt)
                
    is_rd_unchanged = np.all([len(r_to_keep['d'][l]) == r['d'][l] for l in range(L['d'])])
    is_rt_unchanged = np.all([len(r_to_keep['t'][l]) == r['t'][l] for l in range(new_Lt)])
    is_rc_unchanged = np.all([len(r_to_keep['c'][l]) == r['c'][l] for l in range(L['c'] + 1)])
    is_r_unchanged = np.all([is_rd_unchanged, is_rc_unchanged, is_rt_unchanged])
    
    is_kd_unchanged = np.all([len(k_to_keep['d'][l]) == k['d'][l] for l in range(L['d'])])
    is_kt_unchanged = np.all([len(k_to_keep['t'][l]) == k['t'][l] for l in range(new_Lt)])
    is_kc_unchanged = np.all([len(k_to_keep['c'][l]) == k['c'][l] for l in range(L['c'] + 1)])
    is_k_unchanged = np.all([is_kd_unchanged, is_kc_unchanged, is_kt_unchanged])
                
    is_selection = not(is_r_unchanged & is_k_unchanged & is_L_unchanged)
    
    return is_selection


def dgmm_coeff_selection(eta_c_, H_c_, psi_c_, eta_d_, H_d_, psi_d_,\
                         L, r_to_keep, k_to_keep):
    ''' Utility to update the selected components and dimensions of the network
    eta_*_ (list of nb_layers elements of shape (K_l x r_{l-1}, 1)): eta  
                        estimators at the end of the current iteration for 
                        each layer of head *
    H_*_ (list of nb_layers elements of shape (K_l x r_l-1, r_l)): Lambda 
                        estimators at the end of the current iteration for 
                        each layer of head *
    psi_*_ (list of nb_layers elements of shape (K_l x r_l-1, r_l-1)): Psi 
                        estimators at the end of the current iteration for 
                        each layer of head *
    L (dict): The number of layers in the networks 
    r_to_keep (dict): The dimensions selected in the network
    k_to_keep (dict): The components selected in the network
    --------------------------------------------------------------------------
    returns (tuple of size 6): the eta, Lambda and Psi estimators of all 
                                network layers
    '''
     
    #===============================================================
    # Computation of eta
    #===============================================================
    
    # Select the right components and dimensions for both heads 
    eta_c_new = [eta_c_[l][k_to_keep['c'][l]] for l in range(L['c'] + 1)]
    eta_c_new = [eta_c_new[l][:, r_to_keep['c'][l]] for l in range(L['c'] + 1)]
    
    eta_d_new = [eta_d_[l][k_to_keep['d'][l]] for l in range(L['d'])]
    eta_d_new = [eta_d_new[l][:, r_to_keep['d'][l]] for l in range(L['d'])]

    # Select the right components and dimensions for the tail     
    eta_t = [eta_c_[l + L['c'] + 1][k_to_keep['t'][l]] for l in range(L['t'] - 1)]
    eta_t = [eta_t[l][:, r_to_keep['t'][l]] for l in range(L['t'] - 1)]

    # Copy the tail components to the both heads
    eta_c = eta_c_new + eta_t    
    eta_d = eta_d_new + eta_t
    
    #===============================================================
    # Computation of Lambda
    #===============================================================
    
    # Select the right components and dimensions for both heads 
    # Continuous head
    H_c_new = [H_c_[l][k_to_keep['c'][l]] for l in range(L['c'] + 1)]
    H_c_new = [H_c_new[l][:, r_to_keep['c'][l]] for l in range(L['c'] + 1)]
    
    # Part to check
    if L['c'] > 0:
        H_c_new = [H_c_new[l][:, :, r_to_keep['c'][l + 1]] for l in range(L['c'] + 1)]
    
    # For the component between the last head layer and the first tail layer
    H_c_new[-1] = H_c_new[-1][:, :, r_to_keep['t'][0]] 
    
    # Discrete head
    H_d_new = [H_d_[l][k_to_keep['d'][l]] for l in range(L['d'])]
    H_d_new = [H_d_new[l][:, r_to_keep['d'][l]] for l in range(L['d'])]
    
    if L['d'] > 1:
        H_d_new = [H_d_new[l][:, :, r_to_keep['d'][l + 1]] for l in range(L['d'])]
        
    # For the component between the last head layer and the first tail layer
    H_d_new[-1] = H_d_new[-1][:, :, r_to_keep['t'][0]] 
 
    # Common tail
    H_t = [H_c_[l + L['c'] + 1][k_to_keep['t'][l]] for l in range(L['t'] - 1)]
    H_t = [H_t[l][:, r_to_keep['t'][l]] for l in range(L['t'] - 1)]
    H_t = [H_t[l][:, :, r_to_keep['t'][l + 1]] for l in range(L['t'] - 1)]
    
    # Copy the tail components to the both heads
    H_c = H_c_new + H_t    
    H_d = H_d_new + H_t
    
    #===============================================================
    # Computation of psi
    #===============================================================
    psi_c_new = [psi_c_[l][k_to_keep['c'][l]] for l in range(L['c'] + 1)]
    psi_c_new = [psi_c_new[l][:, r_to_keep['c'][l]] for l in range(L['c'] + 1)]
    psi_c_new = [psi_c_new[l][:, :, r_to_keep['c'][l]] for l in range(L['c'] + 1)]
    
    psi_d_new = [psi_d_[l][k_to_keep['d'][l]] for l in range(L['d'])]
    psi_d_new = [psi_d_new[l][:, r_to_keep['d'][l]] for l in range(L['d'])]
    psi_d_new = [psi_d_new[l][:, :, r_to_keep['d'][l]] for l in range(L['d'])]
    
    psi_t = [psi_c_[l + L['c'] + 1][k_to_keep['t'][l]] for l in range(L['t'] - 1)]
    psi_t = [psi_t[l][:, r_to_keep['t'][l]] for l in range(L['t'] - 1)]
    psi_t = [psi_t[l][:, :, r_to_keep['t'][l]] for l in range(L['t'] - 1)]

    # Copy the tail components to the both heads
    psi_c = psi_c_new + psi_t    
    psi_d = psi_d_new + psi_t        
        
    return eta_c, eta_d, H_c, H_d, psi_c, psi_d


def gllvm_coeff_selection(lambda_bin_, lambda_ord_, lambda_categ_, r, r_to_keep):
    ''' Utility to update the selected components and dimensions of the 
        GLLVM coefficients
        
    lambda_bin_ (list of nb_bin x (nj_bin + r1) elements): The binomial coefficients
                                                    at the end of the current iteration
    lambda_ord_ (list of nb_ord ndarrays): The ordinal coefficients
                                                    at the end of the current iteration
    lambda_categ_ (list of nb_ord ndarrays): The ordinal coefficients
                                                    at the end of the current iteration
    r (dict): The original dimensions of the network layers
    r_to_keep (dict): The dimensions selected in the network
    --------------------------------------------------------------------------
    returns (tuple of size 3): the lambda_bin, lambda_ord and lambda_categ estimators 
                                of the first discrete head layer
    '''
    
    nb_bin = len(lambda_bin_)
    nb_ord = len(lambda_ord_)
    nb_categ = len(lambda_categ_)

    
    if nb_bin > 0:
        # Add the intercept:
        bin_r_to_keep = np.concatenate([[0], np.array(r_to_keep['d'][0]) + 1]) 
        lambda_bin = lambda_bin_[:, bin_r_to_keep]
    else:
        lambda_bin = []
     
    if nb_ord > 0:
        # Intercept coefficients handling is a little more complicated here
        lambda_ord_intercept = [lambda_ord_j[:-r['d'][0]] for lambda_ord_j in lambda_ord_]
        Lambda_ord_var = np.stack([lambda_ord_j[-r['d'][0]:] for lambda_ord_j in lambda_ord_])
        Lambda_ord_var = Lambda_ord_var[:, r_to_keep['d'][0]]
        lambda_ord = [np.concatenate([lambda_ord_intercept[j], Lambda_ord_var[j]])\
                      for j in range(nb_ord)]
    else:
        lambda_ord = []
        
    if nb_categ > 0:
        lambda_categ_intercept = [lambda_categ_[j][:, 0]  for j in range(nb_categ)]
        Lambda_categ_var = [lambda_categ_j[:,-r['d'][0]:] for lambda_categ_j in lambda_categ_]
        Lambda_categ_var = [lambda_categ_j[:, r_to_keep['d'][0]] for lambda_categ_j in lambda_categ_]

        lambda_categ = [np.hstack([lambda_categ_intercept[j][..., n_axis], Lambda_categ_var[j]])\
                       for j in range(nb_categ)]  
    else:
        lambda_categ = []
            
    return lambda_bin, lambda_ord, lambda_categ


def path_proba_selection(w_s_c_, w_s_d_, k, k_to_keep, new_Lt):
    ''' Utility to update the path probabilities after the selection
    w_s_* (list): The path probabilities starting from the * head
    k (dict): The original number of component on each layer
    k_to_keep (dict): The components selected in the network
    new_Lt (int): The selected number of layers on the common tail.
    --------------------------------------------------------------------------
    returns (tuple of size 2): The paths probabilities starting from each head
    '''
    
    # Deal with both heads
    w = {'d':  w_s_d_.reshape(*np.concatenate([k['d'], k['t']]), order = 'C'),\
         'c':  w_s_c_.reshape(*np.concatenate([k['c'], k['t']]), order = 'C')}
        
    for h in ['c', 'd']:
        original_Lh = len(w[h].shape)
        new_Lh = len(k[h]) + new_Lt
        
        k_to_keep_ht = k_to_keep[h][:new_Lt] + k_to_keep['t']
        assert (len(k_to_keep_ht) == new_Lh)
        
        new_k_idx_grid = np.ix_(*k_to_keep_ht)
                
        # If layer deletion, sum the last components of the paths
        # Not checked
        if original_Lh > new_Lh: 
            deleted_dims = tuple(range(original_Lh)[new_Lt:])
            w_s = w[h][new_k_idx_grid].sum(deleted_dims).flatten(order = 'C')
        else:
            w_s = w[h][new_k_idx_grid].flatten(order = 'C')
        
        # Renormalization
        w_s /= w_s.sum()
        w[h] = w_s
        
    w_s_c = w['c']
    w_s_d = w['d']
                
    return w_s_c, w_s_d