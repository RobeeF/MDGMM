# -*- coding: utf-8 -*-
"""
Created on Fri May 22 13:07:58 2020

@author: rfuchs
"""

from utilities import compute_path_params
from copy import deepcopy
from numeric_stability import ensure_psd
import autograd.numpy as np
from autograd.numpy import newaxis as n_axis
from autograd.numpy import transpose as t
from autograd.numpy.linalg import cholesky, pinv, eigh
import warnings


def compute_z_moments(w_s, mu_s, sigma_s):
    ''' Compute the first moment and the variance of the latent variable 
    w_s (list of length s1): The path probabilities for all s in S1
    mu_s (list of nd-arrays): The means of the Gaussians starting at each layer
    sigma_s (list of nd-arrays): The covariance matrices of the Gaussians starting at each layer
    -------------------------------------------------------------------------
    returns (tuple of length 2): E(z^{(1)}) and Var(z^{(1)})  
    '''
    full_paths_proba = w_s[..., n_axis, n_axis]
    
    muTmu = mu_s[0] @ t(mu_s[0], (0, 2, 1)) 
    E_z1z1T = (full_paths_proba * (sigma_s[0] + muTmu)).sum(0, keepdims = True)
    Ez1 = (full_paths_proba * mu_s[0]).sum(0, keepdims = True)
    
    var_z1 = E_z1z1T - Ez1 @ t(Ez1, (0,2,1)) 
    try:
        var_z1 = ensure_psd([var_z1])[0] # Numeric stability check
    except:
        #print(var_z1)
        raise RuntimeError('Var z1 was not psd')
    AT = cholesky(var_z1)

    return Ez1, AT


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
        B = np.transpose(H_old[l], (0, 2, 1)) @ pinv(psi_old[l]) @ H_old[l]
        values, vec  = eigh(B)
        H.append(H_old[l] @ vec)
    return H



def identifiable_estim_DDGMM(eta_old, H_old, psi_old, Ez1, AT):
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

    eta_new = deepcopy(eta_old)
    H_new = deepcopy(H_old)
    psi_new = deepcopy(psi_old)
        
    inv_AT = pinv(AT) 
    
    # Identifiability 
    psi_new[0] = inv_AT @ psi_old[0] @ t(inv_AT, (0, 2, 1))
    H_new[0] = inv_AT @ H_old[0]
    eta_new[0] = inv_AT @ (eta_old[0] -  Ez1)    
    
    return eta_new, H_new, psi_new

def head_identifiability(eta, H, psi, w_s):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("error")
        H = diagonal_cond(H, psi)

    mu_s, sigma_s = compute_path_params(eta, H, psi)        
    Ez1, AT = compute_z_moments(w_s, mu_s, sigma_s)
    eta, H, psi = identifiable_estim_DDGMM(eta, H, psi, Ez1, AT)
    return eta, H, psi, AT