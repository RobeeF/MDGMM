# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:52:28 2020

@author: RobF
"""

from copy import deepcopy

from numeric_stability import ensure_psd
from parameter_selection import r_select, k_select 

from identifiability_DGMM import identifiable_estim_DDGMM, compute_z_moments,\
    diagonal_cond
                         
from MCEM_DGMM import draw_z_s, fz2_z1s, draw_z2_z1s, fz_ys,\
    E_step_DGMM_d, M_step_DGMM, draw_z_s_all_network, draw_z2_z1s_network,\
        continuous_lik, fz_s, fz_yCyDs, fy_zs_c, E_step_DGMM_c, E_step_DGMM_t,\
            M_step_DGMM_t, fst_yCyD

from MCEM_GLLVM import draw_zl1_ys, fy_zl1, E_step_GLLVM, \
        bin_params_GLLVM, ord_params_GLLVM
  
from hyperparameters_selection import M_growth, look_for_simpler_network
from utilities import compute_path_params, compute_chsi, compute_rho


import autograd.numpy as np
from autograd.numpy import transpose as t
from autograd.numpy import newaxis as n_axis

from sklearn.preprocessing import StandardScaler


def MDGMM(y, n_clusters, r, k, init, var_distrib, nj, it = 50, \
          eps = 1E-05, maxstep = 100, seed = None, perform_selec = True): 
    
    ''' Fit a Generalized Linear Mixture of Latent Variables Model (GLMLVM)
    
    y (numobs x p ndarray): The observations containing categorical variables
    n_clusters (int): The number of clusters to look for in the data
    r (list): The dimension of latent variables through the first 2 layers
    k (list): The number of components of the latent Gaussian mixture layers
    init (dict): The initialisation parameters for the algorithm
    var_distrib (p 1darray): An array containing the types of the variables in y 
    nj (p 1darray): For binary/count data: The maximum values that the variable can take. 
                    For ordinal data: the number of different existing categories for each variable
    it (int): The maximum number of MCEM iterations of the algorithm
    eps (float): If the likelihood increase by less than eps then the algorithm stops
    maxstep (int): The maximum number of optimisation step for each variable
    seed (int): The random state seed to set (Only for numpy generated data for the moment)
    perform_selec (Bool): Whether to perform architecture selection or not
    ------------------------------------------------------------------------------------------------
    returns (dict): The predicted classes, the likelihood through the EM steps
                    and a continuous representation of the data
    '''

    prev_lik = - 1E12
    best_lik = -1E12
    tol = 0.01
    max_patience = 2
    patience = 0
    
    #====================================================
    # Initialize the parameters
    #====================================================
    
    eta_c = deepcopy(init['c']['eta'])
    eta_d = deepcopy(init['d']['eta'])

    H_c = deepcopy(init['c']['H'])
    H_d = deepcopy(init['d']['H'])

    psi_c = deepcopy(init['c']['psi'])
    psi_d = deepcopy(init['d']['psi'])
        
    lambda_bin = deepcopy(init['lambda_bin'])
    lambda_ord = deepcopy(init['lambda_ord'])
    
    w_s_c = deepcopy(init['c']['w_s']) 
    w_s_d = deepcopy(init['d']['w_s'])
   
    numobs = len(y)
    likelihood = []
    it_num = 0
    ratio = 1000
    np.random.seed = seed

    #====================================================        
    # Dispatch variables between categories
    #====================================================

    y_bin = y[:, np.logical_or(var_distrib == 'bernoulli',\
                               var_distrib == 'binomial')]
    nj_bin = nj[np.logical_or(var_distrib == 'bernoulli',\
                              var_distrib == 'binomial')]
        
    nj_bin = nj_bin.astype(int)
    nb_bin = len(nj_bin)
        
    y_ord = y[:, var_distrib == 'ordinal']    
    nj_ord = nj[var_distrib == 'ordinal']
    nj_ord = nj_ord.astype(int)
    nb_ord = len(nj_ord)
    
    yc = y[:, var_distrib == 'continuous'] 
    
    ss = StandardScaler()
    yc = ss.fit_transform(yc)

    nb_cont = yc.shape[1]
    
    bar_L = {'c': len(k['c']), 'd': len(k['d'])}
        
    # *_1L standsds for quantities going through all the network (head + tail)
    # k and r from head to tail 
    k_1L = {'c': k['c'] + k['t'], 'd': k['d'] + k['t'], 't': k['t']}
    r_1L = {'c': r['c'] + r['t'], 'd': r['d'] + r['t'], 't': r['t']}
    
    # Number of hidden layers of both (heads + tail) and tail
    L_1L = {'c': len(k['c']) + len(k['t']) - 1, 'd': len(k['d']) + len(k['t']), 't': len(k['t'])}
    L = {'c': len(k['c']) - 1, 'd': len(k['d']), 't': len(k['t'])}

    # Paths of both (heads+tail) and tail
    S1cL = [np.prod(k_1L['c'][l:]) for l in range(L_1L['c'] + 1)]
    S1dL = [np.prod(k_1L['d'][l:]) for l in range(L_1L['d'])]
    St = [np.prod(k['t'][l:]) for l in range(L_1L['t'])]
    S_1L = {'c': S1cL, 'd': S1dL, 't': St}
             
    M = M_growth(1, r_1L, numobs) 

    if nb_bin + nb_ord == 0: # Create the InputError class and change this
        raise ValueError('Input does not contain discrete variables,\
                         consider using a regular DGMM')
    if nb_cont == 0: # Create the InputError class and change this
        raise ValueError('Input does not contain continuous values,\
                         consider using a DDGMM')
                         
    # Add assertion about k and r size here
                     
    while (it_num < it) & ((ratio > eps) | (patience <= max_patience)):
        print(it_num)

        # The clustering layer is the one used to perform the clustering 
        # i.e. the layer l such that k[l] == n_clusters
        assert (np.array(k['t']) == n_clusters).any()
        clustering_layer = np.argmax(np.array(k['t']) == n_clusters)

        #####################################################################################
        ################################# MC step ############################################
        #####################################################################################

        #=====================================================================
        # Draw from f(z^{l} | s, Theta) for both heads and tail
        #=====================================================================  
        
        mu_s_c, sigma_s_c = compute_path_params(eta_c, H_c, psi_c)
        sigma_s_c = ensure_psd(sigma_s_c)
        
        mu_s_d, sigma_s_d = compute_path_params(eta_d, H_d, psi_d)
        sigma_s_d = ensure_psd(sigma_s_d)
                
        z_s_c, zc_s_c, z_s_d, zc_s_d = draw_z_s_all_network(mu_s_c, sigma_s_c,\
                            mu_s_d, sigma_s_d, yc, eta_c, eta_d, S_1L, L, M)
        
        #========================================================================
        # Draw from f(z^{l+1} | z^{l}, s, Theta) for l >= 1
        #========================================================================
        
        # Create wrapper as before and after
        chsi_c = compute_chsi(H_c, psi_c, mu_s_c, sigma_s_c)
        chsi_c = ensure_psd(chsi_c)
        rho_c = compute_rho(eta_c, H_c, psi_c, mu_s_c, sigma_s_c, zc_s_c, chsi_c)
        
        chsi_d = compute_chsi(H_d, psi_d, mu_s_d, sigma_s_d)
        chsi_d = ensure_psd(chsi_d)
        rho_d = compute_rho(eta_d, H_d, psi_d, mu_s_d, sigma_s_d, zc_s_d, chsi_d)


        # In the following z2 and z1 will denote z^{l+1} and z^{l} respectively
        z2_z1s_c, z2_z1s_d = draw_z2_z1s_network(chsi_c, chsi_d, rho_c, \
                                                 rho_d, M, r_1L, L)

        #=======================================================================
        # Compute the p(y^D| z1) for all discrete variables
        #=======================================================================
        
        py_zl1_d = fy_zl1(lambda_bin, y_bin, nj_bin, lambda_ord, y_ord, nj_ord, z_s_d[0])
        
        #========================================================================
        # Draw from p(z1 | y, s) proportional to p(y | z1) * p(z1 | s) for all s
        #========================================================================
                
        zl1_ys_d = draw_zl1_ys(z_s_d, py_zl1_d, M['d'])
                
        #####################################################################################
        ################################# E step ############################################
        #####################################################################################
        
        #=====================================================================
        # Compute quantities necessary for E steps of both heads and tail
        #=====================================================================
        
        # Discrete head quantities
        pzl1_ys_d, ps_y_d, py_d = E_step_GLLVM(z_s_d[0], mu_s_d[0], sigma_s_d[0], w_s_d, py_zl1_d)        
        py_s_d = ps_y_d * py_d / w_s_d[n_axis]
        #del(py_zl1)

        
        # Continuous head quantities
        ps_y_c, py_s_c, py_c = continuous_lik(yc, mu_s_c[0], sigma_s_c[0], w_s_c)
        print('p(y^C) = ', np.log(py_c).sum())
        print('p(y^D) = ', np.log(py_d).sum())
        
        pz_s_c = fz_s(z_s_c, mu_s_c, sigma_s_c) 

        #=====================================================================
        # Compute p(z^{(l)}| s, y). Equation (5) of the paper
        #=====================================================================
        
        pz2_z1s_d = fz2_z1s(t(pzl1_ys_d, (1, 0, 2)), z2_z1s_d, chsi_d, rho_d, S_1L['d'])
        pz_ys_d = fz_ys(t(pzl1_ys_d, (1, 0, 2)), pz2_z1s_d)
          
        pz2_z1s_c = fz2_z1s([], z2_z1s_c, chsi_c, rho_c, S_1L['c'])
        pz_ys_c = fz_ys([], pz2_z1s_c)
                
        # Junction layer computations
        # Compute p(zC |s)
        py_zs_c = fy_zs_c(pz_ys_c, py_s_c, pz_s_c)
 
        # Compute p(zt | yC, yD, sC, SD)        
        pzt_yCyDs = fz_yCyDs(py_zs_c, pz_ys_d, py_s_c, L)

        #=====================================================================
        # Compute MFA expectations
        #=====================================================================
        
        # Discrete head. 
        # Pas checkée mais rien ne diffère..?
        Ez_ys_d, E_z1z2T_ys_d, E_z2z2T_ys_d, EeeT_ys_d = \
            E_step_DGMM_d(zl1_ys_d, H_d, z_s_d, zc_s_d, z2_z1s_d, pz_ys_d,\
                        pz2_z1s_d, S_1L['d'], L['d'])
            
        # Continuous head
        Ez_ys_c, E_z1z2T_ys_c, E_z2z2T_ys_c, EeeT_ys_c = \
            E_step_DGMM_c(H_c, z_s_c, zc_s_c, z2_z1s_c, pz_ys_c,\
                          pz2_z1s_c, S_1L['c'], L['c'])

        # Restart from here !!!!                    
        # Junction layers
        Ez_ys_t, E_z1z2T_ys_t, E_z2z2T_ys_t, EeeT_ys_t = \
            E_step_DGMM_t(H_c[bar_L['c']:], \
            z_s_c[bar_L['c']:], zc_s_c[bar_L['c']:], z2_z1s_c[bar_L['c']:],\
                pzt_yCyDs, pz2_z1s_c[bar_L['c']:], S_1L, L, k_1L)  
              
        pst_yCyD = fst_yCyD(py_s_c, py_s_d, w_s_d, py_d, py_c, k_1L, L)                                  
               
        ###########################################################################
        ############################ M step #######################################
        ###########################################################################
             
        #=======================================================
        # Compute DGMM Parameters 
        #=======================================================
            
        #print('New wc', w_s_c.reshape(*k_1L['c'], order = 'C').sum((0, 1)))    
        #print('New wd', w_s_d.reshape(*k_1L['d'], order = 'C').sum((0)))   
                   

        # Discrete head
        w_s_d = np.mean(ps_y_d, axis = 0)      
        eta_d_barL, H_d_barL, psi_d_barL = M_step_DGMM(Ez_ys_d, E_z1z2T_ys_d, E_z2z2T_ys_d, \
                                        EeeT_ys_d, ps_y_d, H_d, k_1L['d'][:-1],\
                                            L_1L['d'], r_1L['d'])
            
        eta_d[:bar_L['d']] = eta_d_barL
        H_d[:bar_L['d']] = H_d_barL
        psi_d[:bar_L['d']] = psi_d_barL
                
        # Continuous head
        w_s_c = np.mean(ps_y_c, axis = 0)  
        eta_c_barL, H_c_barL, psi_c_barL = M_step_DGMM(Ez_ys_c, E_z1z2T_ys_c, E_z2z2T_ys_c, \
                                        EeeT_ys_c, ps_y_c, H_c, k_1L['c'][:-1],\
                                            L_1L['c'] + 1, r_1L['c'])

        eta_c[:bar_L['c']] = eta_c_barL
        H_c[:bar_L['c']] = H_c_barL
        psi_c[:bar_L['c']] = psi_c_barL            

        # Common tail. Wrong args ..? 
        eta_t, H_t, psi_t = M_step_DGMM_t(Ez_ys_t, E_z1z2T_ys_t, E_z2z2T_ys_t, \
                                        EeeT_ys_t, pst_yCyD, \
                                            H_c[bar_L['c']:], k_1L,\
                                            L_1L, L, r_1L['t'])   

        eta_d[bar_L['d']:] = eta_t
        H_d[bar_L['d']:] = H_t
        psi_d[bar_L['d']:] = psi_t            

        eta_c[bar_L['c']:] = eta_t
        H_c[bar_L['c']:] = H_t
        psi_c[bar_L['c']:] = psi_t            
        
        H_d = diagonal_cond(H_d, psi_d)                   
        H_c = diagonal_cond(H_c, psi_c)

        #=======================================================
        # Identifiability conditions
        #======================================================= 
        
        # Update mu and sigma with new eta, H and Psi values
        ## Discrete head
        mu_s_d, sigma_s_d = compute_path_params(eta_d, H_d, psi_d)        
        Ez1_d, AT_d = compute_z_moments(w_s_d, mu_s_d, sigma_s_d)
        eta_d, H_d, psi_d = identifiable_estim_DDGMM(eta_d, H_d, psi_d, Ez1_d, AT_d)
        
        
        ## Continuous head
        mu_s_c, sigma_s_c = compute_path_params(eta_c, H_c, psi_c)        
        #Ez1_c, AT_c = compute_z_moments(w_s_c, mu_s_c, sigma_s_c)
        #eta_c, H_c, psi_c = identifiable_estim_DDGMM(eta_c, H_c, psi_c, Ez1_c, AT_c)
        
    
        del(Ez1_d)
        #del(Ez1_c)
        #=======================================================
        # Compute GLLVM Parameters
        #=======================================================
        
        # We optimize each column separately as it is faster than all column jointly 
        # (and more relevant with the independence hypothesis)
                
        lambda_bin = bin_params_GLLVM(y_bin, nj_bin, lambda_bin, ps_y_d, \
                    pzl1_ys_d, z_s_d[0], AT_d, tol = tol, maxstep = maxstep)
                 
        lambda_ord = ord_params_GLLVM(y_ord, nj_ord, lambda_ord, ps_y_d, \
                    pzl1_ys_d, z_s_d[0], AT_d, tol = tol, maxstep = maxstep)

        ###########################################################################
        ################## Clustering parameters updating #########################
        ###########################################################################
          
        new_lik = np.sum(np.log(py_d) + np.log(py_c))
        likelihood.append(new_lik)
        ratio = (new_lik - prev_lik)/abs(prev_lik)
        print(likelihood)

        ############################ TO FINISH ####################################
        
        # Refresh the classes only if they provide a better explanation of the data
        if best_lik < new_lik:
            best_lik = deepcopy(prev_lik)
            
            idx_to_sum = tuple(set(range(1, L['t'] + 1)) - set([clustering_layer + 1]))
            psl_y = pst_yCyD.reshape(numobs, *k['t'], order = 'C').sum(idx_to_sum) 

            classes = np.argmax(psl_y, axis = 1) 
            
            # To finish
            z = 0#(pst_yCyD[..., n_axis] * Ez_ys_t[clustering_layer]).sum(1)
            
            '''
            fig = plt.figure(figsize=(8,8))
            plt.scatter(z[:, 0], z[:, 1])
            plt.show()
            '''
            
            best_r = deepcopy(r)
            best_k = deepcopy(k)

        
        if prev_lik < new_lik:
            patience = 0
            M = M_growth(it_num + 2, r_1L, numobs)
        else:
            patience += 1
          
        '''                
        ###########################################################################
        ######################## Parameter selection  #############################
        ###########################################################################
        
        is_not_min_specif = not(np.all(k == [n_clusters]) & np.all(r == [2,1]))
        
        if look_for_simpler_network(it_num) & perform_selec & is_not_min_specif:
            r_to_keep = r_select(y_bin, y_ord, zl1_ys, z2_z1s, w_s)
            
            # If r_l == 0, delete the last l + 1: layers
            new_L = np.sum([len(rl) != 0 for rl in r_to_keep]) - 1 
            
            k_to_keep = k_select(w_s, k, new_L, clustering_layer)
    
            is_L_unchanged = L == new_L
            is_r_unchanged = np.all([len(r_to_keep[l]) == r[l] for l in range(new_L + 1)])
            is_k_unchanged = np.all([len(k_to_keep[l]) == k[l] for l in range(new_L)])
              
            is_selection = not(is_r_unchanged & is_k_unchanged & is_L_unchanged )
            
            assert new_L > 0
            
            if is_selection:           
                
                eta = [eta[l][k_to_keep[l]] for l in range(new_L)]
                eta = [eta[l][:, r_to_keep[l]] for l in range(new_L)]
                
                H = [H[l][k_to_keep[l]] for l in range(new_L)]
                H = [H[l][:, r_to_keep[l]] for l in range(new_L)]
                H = [H[l][:, :, r_to_keep[l + 1]] for l in range(new_L)]
                
                psi = [psi[l][k_to_keep[l]] for l in range(new_L)]
                psi = [psi[l][:, r_to_keep[l]] for l in range(new_L)]
                psi = [psi[l][:, :, r_to_keep[l]] for l in range(new_L)]
                
                if nb_bin > 0:
                    # Add the intercept:
                    bin_r_to_keep = np.concatenate([[0], np.array(r_to_keep[0]) + 1]) 
                    lambda_bin = lambda_bin[:, bin_r_to_keep]
                 
                if nb_ord > 0:
                    # Intercept coefficients handling is a little more complicated here
                    lambda_ord_intercept = [lambda_ord_j[:-r[0]] for lambda_ord_j in lambda_ord]
                    Lambda_ord_var = np.stack([lambda_ord_j[-r[0]:] for lambda_ord_j in lambda_ord])
                    Lambda_ord_var = Lambda_ord_var[:, r_to_keep[0]]
                    lambda_ord = [np.concatenate([lambda_ord_intercept[j], Lambda_ord_var[j]])\
                                  for j in range(nb_ord)]
    
                w = w_s.reshape(*k, order = 'C')
                new_k_idx_grid = np.ix_(*k_to_keep[:new_L])
                
                # If layer deletion, sum the last components of the paths
                if L > new_L: 
                    deleted_dims = tuple(range(L)[new_L:])
                    w_s = w[new_k_idx_grid].sum(deleted_dims).flatten(order = 'C')
                else:
                    w_s = w[new_k_idx_grid].flatten(order = 'C')
    
                w_s /= w_s.sum()
    
                k = [len(k_to_keep[l]) for l in range(new_L)]
                r = [len(r_to_keep[l]) for l in range(new_L + 1)]
                
                k_aug = k + [1]
                S = np.array([np.prod(k_aug[l:]) for l in range(new_L + 1)])    
                L = new_L
                
                patience = 0
                         
            print('New architecture:')
            print('k', k)
            print('r', r)
            print('L', L)
            print('S',S)
            print("w_s", len(w_s))
        '''
        
        prev_lik = deepcopy(new_lik)
        it_num = it_num + 1

    out = dict(likelihood = likelihood, classes = classes, z = z, \
               best_r = best_r, best_k = best_k)
    return(out)

