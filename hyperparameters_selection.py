# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:30:16 2020

@author: rfuchs
"""

import autograd.numpy as np

# To merge with parameter selection


# Old one for tictactoe and breast cancer
def M_growth__(it_nb, r_1L, numobs):
    #''' Function that controls the growth rate of M through the iterations
    #it_num (int): The current iteration number
    #r (list of int): The dimensions of each layer
    #---------------------------------------------------------------------
    #returns (1d-array of int): The number of MC points to sample on each layer
    #'''

    M = {}
    M['c'] = (5 * it_nb * np.array(r_1L['c'])).astype(int)
    M['c'][0] = numobs
    M['d'] = (5 * it_nb * np.array(r_1L['d'])).astype(int)  
    
    return M

# Old one for tictactoe and breast cancer
def M_growth(it_nb, r_1L, numobs):
    #''' Function that controls the growth rate of M through the iterations
    #it_num (int): The current iteration number
    #r (list of int): The dimensions of each layer
    #---------------------------------------------------------------------
    #returns (1d-array of int): The number of MC points to sample on each layer
    #'''

    M = {}
    M['c'] = ((40 / np.log(numobs)) * it_nb * np.sqrt(r_1L['c'])).astype(int)
    M['c'][0] = numobs
    M['d'] = ((40 / np.log(numobs)) * it_nb * np.sqrt(r_1L['d'])).astype(int)
    
    return M

def M_growth_new(it_nb, r_1L, numobs):
    ''' Function that controls the growth rate of M through the iterations
    it_num (int): The current iteration number
    r (list of int): The dimensions of each layer
    ---------------------------------------------------------------------
    returns (1d-array of int): The number of MC points to sample on each layer
    '''
    # Do not increase with the iterations
    M = {}
    M['c'] = (20 * np.array(r_1L['c'])).astype(int)
    M['c'][0] = numobs
    M['d'] = (20 * np.array(r_1L['d'])).astype(int)
    
    return M
    
def look_for_simpler_network(it_num):
    ''' Returns whether or not a new architecture of the network have to be 
    looking for at the current iteration.
    it_num (int): The current iteration number
    -------------------------------------------------------------------------
    returns (Bool): True if a simpler architecture has to be looking for 
                    False otherwise
    '''
    if it_num in [0, 1, 7, 10]:
        return True
    else:
        return False
    
def is_min_architecture_reached(k, r, n_clusters):
    
    # Check that common tail is minimal
    first_layer_k_min =  k['t'][0] == n_clusters # geq or eq ?
    following_layers_k_min = np.all([kk <= 2 for kk in k['t'][1:]])
    is_tail_k_min = first_layer_k_min & following_layers_k_min 
    is_tail_r_min = r['t'] == [2, 1]
    
    is_tail_min = is_tail_k_min & is_tail_r_min 
    
    # Check that the heads are minimal
    is_head_k_min = {'c': True, 'd': True}
    is_head_r_min = {'c': True, 'd': True}

    Lt = len(k['t'])

    for h in ['c', 'd']:
        Lh = len(k[h])
        # If more than one layer on head, then arch is not minimal
        if Lh >= 2:
            is_head_k_min[h] = False
            is_head_r_min[h] = False
            continue

        for l in range(Lh):
            # If all k >= 2
            if k[h][l] > 2:
                is_head_k_min[h] = False
            
            # The first dimension of the continuous dimension is imposed
            if (h == 'd') | (l > 0):
                # k is min if r3 = r2 + 1 = r1 + 2 ...
                if r[h][l] > Lh + Lt - l:
                    is_head_r_min[h] = False
    
    are_heads_min = np.all(list(is_head_k_min.values())) & np.all(list(is_head_r_min.values()))
    
    is_arch_min = are_heads_min & is_tail_min
        
    return is_arch_min