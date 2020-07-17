# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 19:33:27 2020

@author: Utilisateur
"""

import autograd.numpy as np
from autograd.numpy import newaxis as n_axis
from scipy.special import binom
from sklearn.preprocessing import OneHotEncoder
from numeric_stability import log_1plusexp, expit


def log_py_zM_bin_j(lambda_bin_j, y_bin_j, zM, k, nj_bin_j): 
    ''' Compute log p(y_j | zM, s1 = k1) of the jth
    
    lambda_bin_j ( (r + 1) 1darray): Coefficients of the binomial distributions in the GLLVM layer
    y_bin_j (numobs 1darray): The subset containing only the binary/count variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    nj_bin_j (int): The number of possible values/maximum values of the jth binary/count variable
    --------------------------------------------------------------
    returns (ndarray): p(y_j | zM, s1 = k1)
    '''
    M = zM.shape[0]
    r = zM.shape[1]
    numobs = len(y_bin_j)
    
    yg = np.repeat(y_bin_j[np.newaxis], axis = 0, repeats = M)
    yg = yg.astype(np.float)

    nj_bin_j = np.float(nj_bin_j)

    coeff_binom = binom(nj_bin_j, yg).reshape(M, 1, numobs)
    
    eta = np.transpose(zM, (0, 2, 1)) @ lambda_bin_j[1:].reshape(1, r, 1)
    eta = eta + lambda_bin_j[0].reshape(1, 1, 1) # Add the constant
    
    den = nj_bin_j * log_1plusexp(eta)
    num = eta @ y_bin_j[np.newaxis, np.newaxis]  
    log_p_y_z = num - den + np.log(coeff_binom)
    
    return np.transpose(log_p_y_z, (0, 2, 1)).astype(np.float)

def log_py_zM_bin(lambda_bin, y_bin, zM, k, nj_bin):
    ''' Compute sum_j log p(y_j | zM, s1 = k1) of all the binomial data with a for loop
    
    lambda_bin (nb_bin x (r + 1) ndarray): Coefficients of the binomial distributions in the GLLVM layer
    y_bin (numobs x nb_bin ndarray): The subset containing only the binary/count variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    nj_bin (nb_bin x 1d-array): The number of possible values/maximum values of binary/count variables respectively
    --------------------------------------------------------------
    returns (ndarray): The sum_j p(y_j | zM, s1 = k1)
    '''
    log_py_zM = 0
    nb_bin = len(nj_bin)
    for j in range(nb_bin):
        log_py_zM += log_py_zM_bin_j(lambda_bin[j], y_bin[:,j], zM, k, nj_bin[j])
        
    return log_py_zM

def binom_loglik_j(lambda_bin_j, y_bin_j, zM, k, ps_y, p_z_ys, nj_bin_j):
    ''' Compute the expected log-likelihood for each binomial variable y_j
    
    lambda_bin_j ( (r + 1) 1darray): Coefficients of the binomial distributions in the GLLVM layer
    y_bin_j (numobs 1darray): The subset containing only the binary/count variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    ps_y (numobs x k ndarray): p(s_i = k1 | y_i) for all k1 in [1,k] and i in [1,numobs]
    p_z_ys (M x numobs x k ndarray): p(z_i | y_i, s_i = k) for all m in [1,M], k1 in [1,k] and i in [1,numobs]
    nj_bin_j (int): The number of possible values/maximum values of the jth binary/count variable
    --------------------------------------------------------------
    returns (float): E_{zM, s | y, theta}(y_bin_j | zM, s1 = k1)
    ''' 
    log_pyzM_j = log_py_zM_bin_j(lambda_bin_j, y_bin_j, zM, k, nj_bin_j)
    return -np.sum(ps_y * np.sum(p_z_ys * log_pyzM_j, axis = 0))


######################################################################
# Ordinal likelihood functions
######################################################################

def log_py_zM_ord_j(lambda_ord_j, y_oh_j, zM, k, nj_ord_j): 
    ''' Compute log p(y_j | zM, s1 = k1) of each ordinal variable 
    
    lambda_ord_j ( (nj_ord_j + r - 1) 1darray): Coefficients of the ordinal distributions in the GLLVM layer
    y_oh_j (numobs 1darray): The jth ordinal variable in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    nj_ord_j (int): The number of possible values values of the jth ordinal variable
    --------------------------------------------------------------
    returns (ndarray): The p(y_j | zM, s1 = k1) for the jth ordinal variable
    '''    
    r = zM.shape[1]
    M = zM.shape[0]
    epsilon = 1E-1 # Numeric stability

    lambda0 = lambda_ord_j[:(nj_ord_j - 1)]
    Lambda = lambda_ord_j[-r:]
 
    broad_lambda0 = lambda0.reshape((nj_ord_j - 1, 1, 1, 1))
    eta = broad_lambda0 - (np.transpose(zM, (0, 2, 1)) @ Lambda.reshape((1, r, 1)))[np.newaxis]
    
    gamma = expit(eta)
    
    gamma_prev = np.concatenate([np.zeros((1,M, k, 1)), gamma])
    gamma_next = np.concatenate([gamma, np.ones((1,M, k, 1))])
    pi = gamma_next - gamma_prev
    
    pi = np.where(pi <= 0, epsilon, pi)
    pi = np.where(pi >= 1, 1 - epsilon, pi)
    
    yg = np.expand_dims(y_oh_j.T, 1)[..., np.newaxis, np.newaxis] 
    
    log_p_y_z = yg * np.log(np.expand_dims(pi, axis=2)) 
   
    return log_p_y_z.sum((0))

def log_py_zM_ord(lambda_ord, y_ord, zM, k, nj_ord): 
    ''' Compute sum_j log p(y_j | zM, s1 = k1) of all the ordinal data with a for loop
    
    lambda_ord ( nb_ord x (nj_ord_j + r - 1) 1darray): Coefficients of the ordinal distributions in the GLLVM layer
    y_ord (numobs x nb_bin ndarray): The subset containing only the binary/count variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    nj_ord (nb_ord x 1d-array): The number of possible values values of ordinal variables
    --------------------------------------------------------------
    returns (ndarray): The sum_j p(y_j | zM, s1 = k1) for ordinal variables
    '''
    
    nb_ord = y_ord.shape[1]
    enc = OneHotEncoder(categories='auto')

    log_pyzM = 0
    for j in range(nb_ord):
        y_oh_j = enc.fit_transform(y_ord[:,j][..., n_axis]).toarray()
        log_pyzM += log_py_zM_ord_j(lambda_ord[j], y_oh_j, zM, k, nj_ord[j])
        
    return log_pyzM
        

def ord_loglik_j(lambda_ord_j, y_oh_j, zM, k, ps_y, p_z_ys, nj_ord_j):
    ''' Compute the expected log-likelihood for each ordinal variable y_j
    lambda_ord_j ( (nj_ord_j + r - 1) 1darray): Coefficients of the ordinal distributions in the GLLVM layer
    y_oh_j (numobs 1darray): The subset containing only the ordinal variables in the dataset
    zM (M x r x k ndarray): M Monte Carlo copies of z for each component k1 of the mixture
    k (int): The number of components of the mixture
    ps_y (numobs x k ndarray): p(s_i = k1 | y_i) for all k1 in [1,k] and i in [1,numobs]
    p_z_ys (M x numobs x k ndarray): p(z_i | y_i, s_i = k) for all m in [1,M], k1 in [1,k] and i in [1,numobs]
    nj_ord_j (int): The number of possible values of the jth ordinal variable
    --------------------------------------------------------------
    returns (float): E_{zM, s | y, theta}(y_ord_j | zM, s1 = k1)
    ''' 
    log_pyzM_j = log_py_zM_ord_j(lambda_ord_j, y_oh_j, zM, k, nj_ord_j)
    return -np.sum(ps_y * np.sum(np.expand_dims(p_z_ys, axis = 3) * log_pyzM_j, (0,3)))

