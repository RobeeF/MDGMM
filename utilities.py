# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 19:26:07 2020

@author: Utilisateur
"""

import autograd.numpy as np

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
    colors = ['red','green'] # For a 2 classes classification
    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes() 
    
    ax.scatter(zl[:, 0], zl[:, 1] , c = classes,\
                    cmap=matplotlib.colors.ListedColormap(colors))
        
    plt.title("2D Latent space representation of the data") 
    ax.set_xlabel('Latent dimension 1', fontweight ='bold')  
    ax.set_ylabel('Latent dimension 2', fontweight ='bold')  
                
    plt.show()

def plot_3d(zl, classes):
    ''' Plot the 3d latent space representation of the data '''
    colors = ['red','green'] # For a 2 classes classification

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
                        cmap = matplotlib.colors.ListedColormap(colors)) 
      
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
    
def check_inputs(k, r):
    
    # Check k and r are dict
    if not(isinstance(k, dict)):
        raise TypeError('k must be a dict')
    
    if not(isinstance(r, dict)):
        raise TypeError('r must be a dict')
            
    # Check keys == ['c', 'd', 't']
    if len(set(k.keys()) - set(['c', 'd', 't'])) != 0:
        raise ValueError('The keys of k have to be [\'c\', \'d\', \'t\']')
        
    if len(set(r.keys()) - set(['c', 'd', 't'])) != 0:
        raise ValueError('The keys of r have to be [\'c\', \'d\', \'t\']') 
    # Check if exist useless keys ...

    # Check k and r have the same length    
    for h, kh in k.items():
        if len(kh) != len(r[h]):
            raise ValueError('r and k must have the same lengths for each head and tail') 

    # Check valid k and r values model
    # ! Implement isnumeric
    for h, kh in k.items():
        if not(np.all([isnumeric(el) for el in kh])):
            raise ValueError('k values must be numeric') 
        if not(np.all([isnumeric(el) for el in r[h]])):
            raise ValueError('r values must be numeric') 
                       
    # Check identifiable model
    for h, kh in k.items():
        if len(kh) != len(r[h]):
            raise ValueError('r and k must have the same lengths for each head and tail') 
            
    
                    

            
    
    
