# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 08:52:28 2020

@author: RobF
"""

from copy import deepcopy

from numeric_stability import ensure_psd
from parameter_selection import r_select, k_select, check_if_selection, \
    dgmm_coeff_selection, gllvm_coeff_selection, path_proba_selection

from identifiability_DGMM import identifiable_estim_DDGMM, compute_z_moments,\
    diagonal_cond
                         
from MCEM_DGMM import fz2_z1s, fz_ys,E_step_DGMM_d, M_step_DGMM,\
    draw_z_s_all_network, draw_z2_z1s_network, continuous_lik,\
    fz_s, fz_yCyDs, fy_zs_c, E_step_DGMM_c, E_step_DGMM_t,\
    M_step_DGMM_t, fst_yCyD

from MCEM_GLLVM import draw_zl1_ys, fy_zl1, E_step_GLLVM, \
        bin_params_GLLVM, ord_params_GLLVM
  
from hyperparameters_selection import M_growth, look_for_simpler_network, \
    is_min_architecture_reached
from utilities import compute_path_params, compute_chsi, compute_rho, \
    plot_2d, plot_3d, check_inputs

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
    
    check_inputs(k, r)

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

    '''
    print('eta_d', np.abs(eta_d[0]).mean())
    print('H_d', np.abs(H_d[0]).mean())
    print('Psi_d', np.abs(psi_d[0]).mean())    
    print('eta_c', np.abs(eta_c[0]).mean())
    print('H_c', np.abs(H_c[0]).mean())
    print('Psi_c', np.abs(psi_c[0]).mean()) 
    print('eta_t', np.abs(eta_c[-1]).mean())                        
    print('H_t', np.abs(H_c[-1]).mean())
    print('Psi_t', np.abs(psi_c[-1]).mean())
    '''
                       
        
    # Add assertion about k and r size here
    # And about their format
                     
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
        #print('p(y^C) = ', np.log(py_c).sum())
        #print('p(y^D) = ', np.log(py_d).sum())
        
        pz_s_c = fz_s(z_s_c, mu_s_c, sigma_s_c) 

        #=====================================================================
        # Compute p(z^{(l)}| s, y). Equation (5) of the paper
        #=====================================================================
        
        # Compute pz2_z1s_d and pz2_z1s_d for the tail indices whereas it is useless
        
        pz2_z1s_d = fz2_z1s(t(pzl1_ys_d, (1, 0, 2)), z2_z1s_d, chsi_d, rho_d, S_1L['d'])
        pz_ys_d = fz_ys(t(pzl1_ys_d, (1, 0, 2)), pz2_z1s_d)
          
        pz2_z1s_c = fz2_z1s([], z2_z1s_c, chsi_c, rho_c, S_1L['c'])
        pz_ys_c = fz_ys([], pz2_z1s_c)
        
        pz2_z1s_t = fz2_z1s([], z2_z1s_c[bar_L['c']:], chsi_c[bar_L['c']:], \
                            rho_c[bar_L['c']:], S_1L['t'])


        # Junction layer computations
        # Compute p(zC |s)
        py_zs_c = fy_zs_c(pz_ys_c, py_s_c, pz_s_c)
 
        # Compute p(zt | yC, yD, sC, SD)        
        pzt_yCyDs = fz_yCyDs(py_zs_c, pz_ys_d, py_s_c, M, S_1L, L)


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


        # Junction layers
        Ez_ys_t, E_z1z2T_ys_t, E_z2z2T_ys_t, EeeT_ys_t = \
            E_step_DGMM_t(H_c[bar_L['c']:], \
            z_s_c[bar_L['c']:], zc_s_c[bar_L['c']:], z2_z1s_c[bar_L['c']:],\
                pzt_yCyDs, pz2_z1s_t, S_1L, L, k_1L)  

        ''' 
        print('E(zD | y, s) =',  np.abs(Ez_ys_d[0]).mean())
        print('E(z1z2D | y, s) =',  np.abs(E_z1z2T_ys_d[0]).mean())
        print('E(z2z2D | y, s) =',  np.abs(E_z2z2T_ys_d[0]).mean())
        print('E(eeTD | y, s) =',  np.abs(EeeT_ys_d[0]).mean())
        
        print('E(zC | y, s) =', np.abs(Ez_ys_c[0]).mean())
        print('E(z1z2C | y, s) =',  np.abs(E_z1z2T_ys_c[0]).mean())
        print('E(z2z2C | y, s) =',  np.abs(E_z2z2T_ys_c[0]).mean())
        print('E(eeTC | y, s) =',  np.abs(EeeT_ys_c[0]).mean())       
        
        
        print('E(zt | y, s) =',  np.abs(Ez_ys_t[0]).mean())
        print('E(z1z2t | y, s) = ',  np.abs(E_z1z2T_ys_t[0]).mean())
        print('E(z2z2t | y, s) =',  np.abs(E_z2z2T_ys_t[0]).mean())
        print('E(eeTt | y, s) =',  np.abs(EeeT_ys_t[0]).mean()) 
        '''
                 
                
        pst_yCyD = fst_yCyD(py_s_c, py_s_d, w_s_d, py_d, py_c, k_1L, L)                                  
               
        ###########################################################################
        ############################ M step #######################################
        ###########################################################################

        #=======================================================
        # Compute DGMM Parameters 
        #=======================================================
            
        #print('New wc', w_s_c.reshape(*k_1L['c'], order = 'C').sum((0)))    
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
                    

        # Common tail
        eta_t, H_t, psi_t, Ezst_y = M_step_DGMM_t(Ez_ys_t, E_z1z2T_ys_t, E_z2z2T_ys_t, \
                                        EeeT_ys_t, ps_y_c, ps_y_d, pst_yCyD, \
                                            H_c[bar_L['c']:], S_1L, k_1L, \
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
        Ez1_c, AT_c = compute_z_moments(w_s_c, mu_s_c, sigma_s_c)
        eta_c, H_c, psi_c = identifiable_estim_DDGMM(eta_c, H_c, psi_c, Ez1_c, AT_c)
    
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
        #print(likelihood)
    
        # Refresh the classes only if they provide a better explanation of the data
        if best_lik < new_lik:
            best_lik = deepcopy(prev_lik)
            
            idx_to_sum = tuple(set(range(1, L['t'] + 1)) - set([clustering_layer + 1]))
            psl_y = pst_yCyD.reshape(numobs, *k['t'], order = 'C').sum(idx_to_sum) 

            classes = np.argmax(psl_y, axis = 1) 
            
            # To finish
            z_tail = [Ezst_y[l].sum(1) for l in range(L['t'] - 1)]
            z = Ezst_y[clustering_layer].sum(1)
             
            for l in range(L['t'] - 1):
                zl = Ezst_y[l].sum(1)
                if zl.shape[-1] == 3:
                    plot_3d(zl, classes)
                elif zl.shape[-1] == 2:
                    plot_2d(zl, classes)

                    '''
                    colors = ['red','green'] # For a 2 classes classification
                    fig = plt.figure(figsize=(8,8))
                    plt.scatter(zl[:, 0], zl[:, 1] , c = classes,\
                            cmap=matplotlib.colors.ListedColormap(colors))
                    '''
                    
            #pd.DataFrame(psl_y[:,0]).plot(kind='hist') 
            
            best_r = deepcopy(r)
            best_k = deepcopy(k)

        
        if prev_lik < new_lik:
            patience = 0
            M = M_growth(it_num + 1, r_1L, numobs)
        else:
            patience += 1
            
        if ps_y_d.max() > 1:
            print('ps_y_d > 1', ps_y_d.max())
        if ps_y_c.max() > 1:
            print('ps_y_c > 1', ps_y_c.max())  
        if pst_yCyD.max() > 1:
            print('pst_yCyD > 1', pst_yCyD.max())


        '''
        py_zl1_d pzl1_ys_d ps_y_d py_d py_s_d pz_s_c pz2_z1s_d pz_ys_d pz2_z1s_c pz_ys_c pz2_z1s_t
        py_zs_c pzt_yCyDs pst_yCyD
        '''
          
                        
        ###########################################################################
        ######################## Parameter selection  #############################
        ###########################################################################
        
        is_not_min_specif = not(is_min_architecture_reached(k, r, n_clusters))
        
        print('is not min spe', is_not_min_specif)
        print('Look for simpler network', look_for_simpler_network(it_num))
        
        
        if look_for_simpler_network(it_num) & perform_selec & is_not_min_specif:
            
            # Select only Lt for the moment and not Ld and Lc for the layers
            r_to_keep = r_select(y_bin, y_ord, yc, zl1_ys_d,\
                                 z2_z1s_d[:bar_L['d']], w_s_d, z2_z1s_c[:bar_L['c']],
                                 z2_z1s_c[bar_L['c']:])
            
            # If r_l == 0, delete the last l + 1: layers
            new_Lt = np.sum([len(rl) != 0 for rl in r_to_keep['t']]) #- 1
            
            w_s_t = pst_yCyD.mean(0)
            k_to_keep = k_select(w_s_c, w_s_d, w_s_t, k, new_Lt, clustering_layer)
                        
            is_selection = check_if_selection(r_to_keep, r, k_to_keep, k, L, new_Lt)
            
            assert new_Lt > 0
            
            if is_selection:
                
                # Part to change when update also number of layers on each head 
                nb_deleted_layers_tail = L['t'] - new_Lt
                L['t'] = new_Lt
                L_1L = {keys: values - nb_deleted_layers_tail for keys, values in L_1L.items()}
                
                eta_c, eta_d, H_c, H_d, psi_c, psi_d = dgmm_coeff_selection(eta_c,\
                            H_c, psi_c, eta_d, H_d, psi_d, L, r_to_keep, k_to_keep)
                    
                lambda_bin, lambda_ord = gllvm_coeff_selection(lambda_bin, lambda_ord, r, r_to_keep)
                
                # Error here !
                w_s_c, w_s_d = path_proba_selection(w_s_c, w_s_d, k, k_to_keep, new_Lt)
                
                k = {h: [len(k_to_keep[h][l]) for l in range(L[h])] for h in ['d', 't']}
                k['c'] = [len(k_to_keep['c'][l]) for l in range(L['c'] + 1)]
                
                r = {h: [len(r_to_keep[h][l]) for l in range(L[h])] for h in ['d', 't']}
                r['c'] = [len(r_to_keep['c'][l]) for l in range(L['c'] + 1)]
                
                bar_L = {'c': len(k['c']), 'd': len(k['d'])}
        
                k_1L = {'c': k['c'] + k['t'], 'd': k['d'] + k['t'], 't': k['t']}
                r_1L = {'c': r['c'] + r['t'], 'd': r['d'] + r['t'], 't': r['t']}
                
                # Paths of both (heads+tail) and tail
                S1cL = [np.prod(k_1L['c'][l:]) for l in range(L_1L['c'] + 1)]
                S1dL = [np.prod(k_1L['d'][l:]) for l in range(L_1L['d'])]
                St = [np.prod(k['t'][l:]) for l in range(L_1L['t'])]
                S_1L = {'c': S1cL, 'd': S1dL, 't': St}
                

                                            
                patience = 0
                         
            print('New architecture:')
            print('k', k)
            print('r', r)
            print('L', L)
            print('S_1L', S_1L)
            print("w_s_c", len(w_s_c))
            print("w_s_d", len(w_s_d))
        
        # Reinforce the identifiability conditions again ?
        # New M growth ? 
        M = M_growth(it_num + 1, r_1L, numobs)
        
        prev_lik = deepcopy(new_lik)
        it_num = it_num + 1
        print(likelihood)
        '''
        print('eta_d', np.abs(eta_d[0]).mean())
        print('H_d', np.abs(H_d[0]).mean())
        print('Psi_d', np.abs(psi_d[0]).mean())
        print('eta_c', np.abs(eta_c[0]).mean())
        print('H_c', np.abs(H_c[0]).mean())
        print('Psi_c', np.abs(psi_c[0]).mean()) 
        print('eta_t', np.abs(eta_t[0]).mean())                        
        print('H_t', np.abs(H_t[0]).mean())
        print('Psi_t', np.abs(psi_t[0]).mean())
        '''

    out = dict(likelihood = likelihood, classes = classes, z = z_tail[clustering_layer], \
               best_r = best_r, best_k = best_k)
    return(out)

