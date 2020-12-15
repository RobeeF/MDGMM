# -*- coding: utf-8 -*-
"""
Created on Fri May 22 13:07:58 2020

@author: rfuchs
"""

from numeric_stability import ensure_psd
import autograd.numpy as np
from autograd.numpy import newaxis as n_axis
from autograd.numpy import transpose as t
from autograd.numpy.linalg import cholesky, pinv, eigh
import warnings


def compute_z_moments(w_s, eta_old, H_old, psi_old):
    ''' Compute the first moment and the variance of the latent variable 
    w_s (list of length s1): The path probabilities for all s in S1
    eta_old (list of nb_layers elements of shape (K_l x r_{l-1}, 1)): eta  
                        estimators of the previous iteration for each layer
    H_old (list of nb_layers elements of shape (K_l x r_l-1, r_l)): Lambda 
                        estimators of the previous iteration for each layer
    psi_old (list of nb_layers elements of shape (K_l x r_l-1, r_l-1)): Psi 
                        estimators of the previous iteration for each layer
    -------------------------------------------------------------------------
    returns (tuple of length 2): E(z^{(l)}) and Var(z^{(l)})  
    '''
    
    k = [eta.shape[0] for eta in eta_old]
    L = len(eta_old) 
    
    Ez = [[] for l in range(L)]
    AT = [[] for l in range(L)]
    
    w_reshaped = w_s.reshape(*k, order = 'C')
    
    for l in reversed(range(L)):
        # Compute E(z^{(l)})
        idx_to_sum = tuple(set(range(L)) - set([l]))
        
        wl = w_reshaped.sum(idx_to_sum)[..., n_axis, n_axis]
        Ezl = (wl * eta_old[l]).sum(0, keepdims = True)
        Ez[l] = Ezl
        
        etaTeta = eta_old[l] @ t(eta_old[l], (0, 2, 1)) 
        HlHlT = H_old[l] @ t(H_old[l], (0, 2, 1)) 
        
        E_zlzlT = (wl * (HlHlT + psi_old[l] + etaTeta)).sum(0, keepdims = True)
        var_zl = E_zlzlT - Ezl @ t(Ezl, (0,2,1)) 

        try:
            var_zl = ensure_psd([var_zl])[0] # Numeric stability check
        except:
            print(var_zl)
            raise RuntimeError('Var z1 was not psd')        

        AT_l = cholesky(var_zl)
        AT[l] = AT_l

    return Ez, AT

# Function name might be improved:
def identifiable_estim_DDGMM(eta_old, H_old, psi_old, Ez, AT):
    ''' Ensure that the latent variables are centered reduced 
    (1st DGMM identifiability condition)
    
    eta_old (list of nb_layers elements of shape (K_l x r_{l-1}, 1)): mu  
                        estimators of the previous iteration for each layer
    H_old (list of nb_layers elements of shape (K_l x r_l-1, r_l)): Lambda 
                        estimators of the previous iteration for each layer
    psi_old (list of nb_layers elements of shape (K_l x r_l-1, r_l-1)): Psi 
                        estimators of the previous iteration for each layer
    Ez1 (list of (k_l, r_l) ndarray): E(z^{(l)})
    AT (list of (k_l, k_l) ndarray): Var(z^{(l)})^{-1/2 T}
    -------------------------------------------------------------------------
    returns (tuple of length 3): "DDGMM identifiable" estimators of eta, Lambda and Psi
    ''' 


    L = len(eta_old)
    
    eta_new = [[] for l in range(L)]
    H_new = [[] for l in range(L)]
    psi_new = [[] for l in range(L)]
    
    for l in reversed(range(L)):
        inv_AT = pinv(AT[l])

        # Identifiability 
        psi_new[l] = inv_AT @ psi_old[l] @ t(inv_AT, (0, 2, 1))
        H_new[l] = inv_AT @ H_old[l]
        eta_new[l] = inv_AT @ (eta_old[l] -  Ez[l])    
        
    return eta_new, H_new, psi_new


def diagonal_cond(H_old, psi_old):
    ''' Ensure that Lambda^T Psi^{-1} Lambda is diagonal 
    (2nd DGMM identifiability condition)
    
    H_old (list of nb_layers elements of shape (K_l x r_l-1, r_l)): The previous
                                        iteration values of Lambda estimators
    psi_old (list of ndarrays): The previous iteration values of Psi estimators
                    (list of nb_layers elements of shape (K_l x r_l-1, r_l-1))
    ------------------------------------------------------------------------
    returns (list of nb_layers elements of shape (K_l x r_l-1, r_l)): The 
                                                "DGMM identifiable" H estimator
    '''
    
    L = len(H_old)
    
    H = []
    for l in range(L):
        B = np.transpose(H_old[l], (0, 2, 1)) @ pinv(psi_old[l], rcond=1e-3) @ H_old[l]
        values, vec  = eigh(B)
        H.append(H_old[l] @ vec)
    return H


def head_tail_identifiability(eta_old, H_old, psi_old, w_s):
    '''
    Applies the two identifiability conditions to each head layers of the network
    eta_old (list of nb_layers elements of shape (K_l x r_{l-1}, 1)): mu  
                        estimators of the previous iteration for each layer
    H_old (list of nb_layers elements of shape (K_l x r_l-1, r_l)): Lambda 
                        estimators of the previous iteration for each layer
    psi_old (list of nb_layers elements of shape (K_l x r_l-1, r_l-1)): Psi 
                        estimators of the previous iteration for each layer
    w_s (list of length s1): The path probabilities for all s in S^h
    -------------------------------------------------------------------------
    returns (tuple of length 4): "identifiable" estimators of eta, Lambda and Psi
                                and the covariance matrices of each layer latent 
                                variable
    '''
    
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("default")

        H = diagonal_cond(H_old, psi_old)
        Ez, AT = compute_z_moments(w_s, eta_old, H_old, psi_old)
        eta, H, psi = identifiable_estim_DDGMM(eta_old, H_old, psi_old, Ez, AT)
        
    return eta, H, psi, AT

def network_identifiability(eta_d_old, H_d_old, psi_d_old, eta_c_old, H_c_old, 
                            psi_c_old, w_s_c, w_s_d, w_s_t, bar_L):
    ''' Applies the two identifiability conditions on the whole network
    eta_*_old (list of nb_layers elements of shape (K_l x r_{l-1}, 1)): mu  
                        estimators of the previous iteration for each layer 
                        of the given head *
    H_*_old (list of nb_layers elements of shape (K_l x r_l-1, r_l)): Lambda 
                        estimators of the previous iteration for each layer
                        of the given head *
    psi_*_old (list of nb_layers elements of shape (K_l x r_l-1, r_l-1)): Psi 
                        estimators of the previous iteration for each layer 
                        of the given head *
    w_s_* (list of length s1): The path probabilities for all s in S^h 
                        of the given head or tail *
    bar_L (dict): The index of the last head layer for both head
    --------------------------------------------------------------------------
    returns (tuple of length 8): "identifiable" estimators of eta, Lambda and Psi
                                and the covariance matrices of each layer latent 
                                variable for both heads.
    '''
    
    eta_d, H_d, psi_d, AT_d = head_tail_identifiability(eta_d_old, H_d_old, psi_d_old, w_s_d)
    eta_c, H_c, psi_c, AT_c = head_tail_identifiability(eta_c_old, H_c_old, psi_c_old, w_s_c)
    eta_t, H_t, psi_t, AT_t = head_tail_identifiability(eta_c_old[bar_L['c']:], H_c_old[bar_L['c']:],\
                                                   psi_c_old[bar_L['c']:], w_s_t)
        
    eta_d[bar_L['d']:] = eta_t
    H_d[bar_L['d']:] = H_t
    psi_d[bar_L['d']:] = psi_t            
    AT_d[bar_L['d']:] = AT_t            

    eta_c[bar_L['c']:] = eta_t
    H_c[bar_L['c']:] = H_t
    psi_c[bar_L['c']:] = psi_t  
    AT_c[bar_L['c']:] = AT_t  
        
    return eta_d, H_d, psi_d, AT_d, eta_c, H_c, psi_c, AT_c