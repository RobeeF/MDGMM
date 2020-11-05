# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 19:26:07 2020

@author: Utilisateur
"""

from copy import deepcopy
import autograd.numpy as np
from mpl_toolkits.mplot3d import Axes3D

from autograd.numpy.linalg import pinv
from autograd.numpy import newaxis as n_axis
from autograd.numpy import transpose as t


##########################################################################################################
#################################### DGMM Utils ##########################################################
##########################################################################################################

def repeat_tile(x, reps, tiles):
    ''' Repeat then tile a quantity to mimic the former code logic
    reps (int): The number of times to repeat the first axis
    tiles (int): The number of times to tile the second axis
    -----------------------------------------------------------
    returns (ndarray): The repeated then tiled nd_array
    '''
    x_rep = np.repeat(x, reps, axis = 0)
    x_tile_rep = np.tile(x_rep, (tiles, 1, 1))
    return x_tile_rep
        

def compute_path_params(eta, H, psi):
    ''' Compute the gaussian parameters for each path
    H (list of nb_layers elements of shape (K_l x r_{l-1}, r_l)): Lambda 
                                                    parameters for each layer
    psi (list of nb_layers elements of shape (K_l x r_{l-1}, r_{l-1})): Psi 
                                                    parameters for each layer
    eta (list of nb_layers elements of shape (K_l x r_{l-1}, 1)): mu 
                                                    parameters for each layer
    ------------------------------------------------------------------------------------------------
    returns (tuple of len 2): The updated parameters mu_s and sigma for all s in Omega
    '''
    
    #=====================================================================
    # Retrieving model parameters
    #=====================================================================
    
    L = len(H)
    k = [len(h) for h in H]
    k_aug = k + [1] # Integrating the number of components of the last layer i.e 1
    
    r1 = H[0].shape[1]
    r2_L = [h.shape[2] for h in H] # r[2:L]
    r = [r1] + r2_L # r augmented
    
    #=====================================================================
    # Initiating the parameters for all layers
    #=====================================================================
    
    mu_s = [0 for i in range(L + 1)]
    sigma_s = [0 for i in range(L + 1)]
    
    # Initialization with the parameters of the last layer
    mu_s[-1] = np.zeros((1, r[-1], 1)) # Inverser k et r plus tard
    sigma_s[-1] = np.eye(r[-1])[n_axis]
    
    #==================================================================================
    # Compute Gaussian parameters from top to bottom for each path
    #==================================================================================

    for l in reversed(range(0, L)):
        H_repeat = np.repeat(H[l], np.prod(k_aug[l + 1: ]), axis = 0)
        eta_repeat = np.repeat(eta[l], np.prod(k_aug[l + 1: ]), axis = 0)
        psi_repeat = np.repeat(psi[l], np.prod(k_aug[l + 1: ]), axis = 0)
        
        mu_s[l] = eta_repeat + H_repeat @ np.tile(mu_s[l + 1], (k[l], 1, 1))
        
        sigma_s[l] = H_repeat @ np.tile(sigma_s[l + 1], (k[l], 1, 1)) @ t(H_repeat, (0, 2, 1)) \
            + psi_repeat
        
    return mu_s, sigma_s


def compute_chsi(H, psi, mu_s, sigma_s):
    ''' Compute chsi as defined in equation (8) of the DGMM paper 
    H (list of nb_layers elements of shape (K_l x r_l-1, r_l)): Lambda 
                                                    parameters for each layer
    psi (list of nb_layers elements of shape (K_l x r_l-1, r_l-1)): Psi 
                                                    parameters for each layer
    mu_s (list of nd-arrays): The means of the Gaussians starting at each layer
    sigma_s (list of nd-arrays): The covariance matrices of the Gaussians 
                                                    starting at each layer
    ------------------------------------------------------------------------------------------------
    returns (list of ndarray): The chsi parameters for all paths starting at each layer
    '''
    L = len(H)
    k = [len(h) for h in H]
    
    #=====================================================================
    # Initiating the parameters for all layers
    #=====================================================================
    
    # Initialization with the parameters of the last layer    
    chsi = [0 for i in range(L)]
    chsi[-1] = pinv(pinv(sigma_s[-1]) + t(H[-1], (0, 2, 1)) @ pinv(psi[-1]) @ H[-1]) 

    #==================================================================================
    # Compute chsi from top to bottom 
    #==================================================================================
        
    for l in range(L - 1):
        Ht_psi_H = t(H[l], (0, 2, 1)) @ pinv(psi[l]) @ H[l]
        Ht_psi_H = np.repeat(Ht_psi_H, np.prod(k[l + 1:]), axis = 0)
        
        sigma_next_l = np.tile(sigma_s[l + 1], (k[l], 1, 1))
        chsi[l] = pinv(pinv(sigma_next_l) + Ht_psi_H)
            
    return chsi

def compute_rho(eta, H, psi, mu_s, sigma_s, z_c, chsi):
    ''' Compute rho as defined in equation (8) of the DGMM paper 
    eta (list of nb_layers elements of shape (K_l x r_{l-1}, 1)): mu 
                                                    parameters for each layer    
    H (list of nb_layers elements of shape (K_l x r_{l-1}, r_l)): Lambda 
                                                    parameters for each layer
    psi (list of nb_layers elements of shape (K_l x r_{l-1}, r_{l-1})): Psi 
                                                    parameters for each layer
    z_c (list of nd-arrays) z^{(l)} - eta^{(l)} for each layer. 
    chsi (list of nd-arrays): The chsi parameters for each layer
    -----------------------------------------------------------------------
    returns (list of ndarrays): The rho parameters (covariance matrices) 
                                    for all paths starting at each layer
    '''
    
    L = len(H)    
    rho = [0 for i in range(L)]
    k = [len(h) for h in H]
    k_aug = k + [1] 

    for l in range(0, L):
        sigma_next_l = np.tile(sigma_s[l + 1], (k[l], 1, 1))
        mu_next_l = np.tile(mu_s[l + 1], (k[l], 1, 1))

        HxPsi_inv = t(H[l], (0, 2, 1)) @ pinv(psi[l])
        HxPsi_inv = np.repeat(HxPsi_inv, np.prod(k_aug[l + 1: ]), axis = 0)

        rho[l] = chsi[l][n_axis] @ (HxPsi_inv[n_axis] @ z_c[l][..., n_axis] \
                                    + (pinv(sigma_next_l) @ mu_next_l)[n_axis])
                
    return rho


##########################################################################################################
#################################### DGMM Utils ##########################################################
##########################################################################################################

import matplotlib
import matplotlib.pyplot as plt 


def plot_2d(zl, classes):
    ''' Plot the 2d representation of the data '''
    
    n_clusters = len(np.unique(classes))

    colors = ['red', 'green', 'blue', 'silver', 'purple', 'black',\
              'gold', 'orange'] # For a 2 classes classification
    
    if n_clusters >= len(colors):
        raise ValueError('Too many classes for plotting,\
                         please add some colors names above this line')
                         
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes() 
    
    ax.scatter(zl[:, 0], zl[:, 1] , c = classes,\
                    cmap=matplotlib.colors.ListedColormap(colors[:n_clusters]))
        
    plt.title("2D Latent space representation of the data") 
    ax.set_xlabel('Latent dimension 1', fontweight ='bold')  
    ax.set_ylabel('Latent dimension 2', fontweight ='bold')  
                
    plt.show()

def plot_3d(zl, classes):
    ''' Plot the 3d latent space representation of the data '''
    
    n_clusters = len(np.unique(classes))
    colors = ['red', 'green', 'blue', 'silver', 'purple', 'black',\
              'gold', 'orange'] # For a 2 classes classification
    
    if n_clusters >= len(colors):
        raise ValueError('Too many classes for plotting,\
                         please add some colors names above this line')

    fig = plt.figure(figsize = (16, 9)) 
    ax = plt.axes(projection ="3d") 
        
    # Add x, y gridlines  
    ax.grid(b = True, color ='grey',  
            linestyle ='-.', linewidth = 0.3,  
            alpha = 0.2)  
      
    # Creating plot 
    sctt = ax.scatter3D(zl[:,0], zl[:,1], zl[:,2], 
                        alpha = 0.8, 
                        c = classes,  
                        cmap = matplotlib.colors.ListedColormap(colors[:n_clusters])) 
      
    plt.title("3D Latent space representation of the data") 
    ax.set_xlabel('Latent dimension 1', fontweight ='bold')  
    ax.set_ylabel('Latent dimension 2', fontweight ='bold')  
    ax.set_zlabel('Latent dimension 3', fontweight ='bold') 
    #fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5) 
      
    # show plot 
    plt.show() 

##########################################################################################################
################################# General purposes #######################################################
##########################################################################################################
   

def isnumeric(var):
    is_num = False
    try:
        int(var)
        is_num = True
    except:
        pass
    return is_num

def asnumeric(lst):
    try:
        lst = [int(el) for el in lst]
    except:
        raise ValueError('r  and k values must be numeric') 
    return lst

def check_inputs(k, r):
    
    # Check k and r are dict
    if not(isinstance(k, dict)):
        raise TypeError('k must be a dict')
    
    if not(isinstance(r, dict)):
        raise TypeError('r must be a dict')
            
    # Check keys == ['c', 'd', 't']
    if set(k.keys()) != set(['c', 'd', 't']):
        raise ValueError('The keys of k have to be [\'c\', \'d\', \'t\']')
        
    if set(r.keys()) != set(['c', 'd', 't']):
        raise ValueError('The keys of r have to be [\'c\', \'d\', \'t\']') 

    # Check k and r have the same length    
    for h in ['c', 'd', 't']:
        if len(k[h]) != len(r[h]):
            raise ValueError('r and k must have the same lengths for each head and tail') 

    # Check valid k and r values model
    # ! Implement isnumeric
    for h, kh in k.items():
        k[h] = asnumeric(kh)
        r[h] =  asnumeric(r[h])
        
    # Check k['c'] is 1
    if k['c'][0] != 1:
        raise ValueError('The first continuous head layer are the data hence k[\'c\'] = 1')
        
    # Check identifiable model
    for h in ['c', 'd']:
        r_1Lh =  r[h] + r['t']
        are_dims_decreasing = np.all([r_1Lh[l] - r_1Lh[l + 1] > 0 \
                                     for l in range(len(r_1Lh) - 1)])
        if not(are_dims_decreasing):
            raise ValueError('Dims must be decreasing from heads to tail !')
        
###############################################################################
############################ Syntaxic Sugar ###################################
###############################################################################
                
def dispatch_dgmm_init(init):
    eta_c = deepcopy(init['c']['eta'])
    eta_d = deepcopy(init['d']['eta'])

    H_c = deepcopy(init['c']['H'])
    H_d = deepcopy(init['d']['H'])

    psi_c = deepcopy(init['c']['psi'])
    psi_d = deepcopy(init['d']['psi'])
    
    return eta_c, eta_d, H_c, H_d, psi_c, psi_d

def dispatch_gllvm_init(init):
    lambda_bin = deepcopy(init['lambda_bin'])
    lambda_ord = deepcopy(init['lambda_ord'])
    lambda_categ = deepcopy(init['lambda_categ'])
    
    return lambda_bin, lambda_ord, lambda_categ

def dispatch_paths_init(init):
    w_s_c = deepcopy(init['c']['w_s']) 
    w_s_d = deepcopy(init['d']['w_s'])
    return w_s_c, w_s_d

def compute_S_1L(L_1L, k_1L, k):
    # Paths of both (heads+tail) and tail
    S1cL = [np.prod(k_1L['c'][l:]) for l in range(L_1L['c'] + 1)]
    S1dL = [np.prod(k_1L['d'][l:]) for l in range(L_1L['d'])]
    St = [np.prod(k['t'][l:]) for l in range(L_1L['t'])]
    return {'c': S1cL, 'd': S1dL, 't': St}
    

def nb_comps_and_layers(k):
    k_1L = {'c': k['c'] + k['t'], 'd': k['d'] + k['t'], 't': k['t']}
    
    # Number of hidden layers of both (heads + tail) and tail
    L_1L = {'c': len(k['c']) + len(k['t']) - 1, 'd': len(k['d']) + len(k['t']),\
            't': len(k['t'])}
    L = {'c': len(k['c']) - 1, 'd': len(k['d']), 't': len(k['t'])}
    bar_L = {'c': len(k['c']), 'd': len(k['d'])}

    S_1L = compute_S_1L(L_1L, k_1L, k)
    return k_1L, L_1L, L, bar_L, S_1L
