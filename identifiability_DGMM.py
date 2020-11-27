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
    mu_s (list of nd-arrays): The means of the Gaussians starting at each layer
    sigma_s (list of nd-arrays): The covariance matrices of the Gaussians starting at each layer
    -------------------------------------------------------------------------
    returns (tuple of length 2): E(z^{(1)}) and Var(z^{(1)})  
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


def identifiable_estim_DDGMM(eta_old, H_old, psi_old, Ez, AT):
    ''' Enforce identifiability conditions for DGMM estimators
    eta_old (list of nb_layers elements of shape (K_l x r_{l-1}, 1)): mu  
                        estimators of the previous iteration for each layer
    H_old (list of nb_layers elements of shape (K_l x r_l-1, r_l)): Lambda 
                        estimators of the previous iteration for each layer
    psi (list of nb_layers elements of shape (K_l x r_l-1, r_l-1)): Psi 
                        estimators of the previous iteration for each layer
    Ez1 ((1, k1, 1) ndarray): E(z^{(1)})
    AT ((1, k1, k1) ndarray): Var(z^{(1)})^{-1/2 T}
    -------------------------------------------------------------------------
    returns (tuple of length 3): "identifiable" esimators of eta, Lambda and Psi
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


'''
H_old = H_c
psi_old = psi_c
'''

def diagonal_cond(H_old, psi_old):
    ''' Ensure that Lambda^T Psi^{-1} Lambda is diagonal
    H_old (list of nb_layers elements of shape (K_l x r_l-1, r_l)): The previous
                                        iteration values of Lambda estimators
    psi_old (list of ndarrays): The previous iteration values of Psi estimators
                    (list of nb_layers elements of shape (K_l x r_l-1, r_l-1))
    ------------------------------------------------------------------------
    returns (list of nb_layers elements of shape (K_l x r_l-1, r_l)): The 
                                                "identifiable" H estimator
    '''
    L = len(H_old)
    
    H = []
    for l in range(L):
        B = np.transpose(H_old[l], (0, 2, 1)) @ pinv(psi_old[l], rcond=1e-3) @ H_old[l]
        values, vec  = eigh(B)
        H.append(H_old[l] @ vec)
    return H


def head_tail_identifiability(eta_old, H_old, psi_old, w_s):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("default")

        H = diagonal_cond(H_old, psi_old)
        Ez, AT = compute_z_moments(w_s, eta_old, H_old, psi_old)
        eta, H, psi = identifiable_estim_DDGMM(eta_old, H_old, psi_old, Ez, AT)
        
    return eta, H, psi, AT

'''
eta_d_old = deepcopy(eta_d)
H_d_old = deepcopy(H_d)
psi_d_old = deepcopy(psi_d)

eta_c_old = deepcopy(eta_c)
H_c_old = deepcopy(H_c)
psi_c_old = deepcopy(psi_c)

'''


def network_identifiability(eta_d_old, H_d_old, psi_d_old, eta_c_old, H_c_old, psi_c_old,\
                            w_s_c, w_s_d, w_s_t, bar_L):
    ''' Ensure that the network is identified '''
    
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