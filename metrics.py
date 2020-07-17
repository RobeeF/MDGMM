# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:13:40 2020

@author: rfuchs
"""

from copy import deepcopy
from itertools import permutations

import autograd.numpy as np

def misc(true, pred, return_relabeled = False):
    ''' Computes a label invariant misclassification error and can return the 
    relabeled predictions (the one that results in the least misc compared 
    to the original labels).
    true (numobs 1darray): array with the true labels
    pred (numobs 1darray): array with the predicted labels
    return_relabeled (Bool): Whether or not to return the relabeled predictions
    --------------------------------------------------------
    returns (float): The misclassification error rate  
    '''
    best_misc = 0
    true_classes = np.unique(true).astype(int)
    nb_classes = len(true_classes)
    
    best_labeled_pred = pred

    best_misc = 1
    
    # Compute of the possible labelling
    all_possible_labels = [list(l) for l in list(permutations(true_classes))]
    
    # And compute the misc for each labelling
    for l in all_possible_labels:
        shift = max(true_classes) + 1
        shift_pred = pred + max(true_classes) + 1
        
        for i in range(nb_classes):
            shift_pred = np.where(shift_pred == i + shift, l[i], shift_pred)
        
        current_misc = np.mean(true != shift_pred)
        if current_misc < best_misc:
            best_misc = deepcopy(current_misc)
            best_labeled_pred = deepcopy(shift_pred)
      
    if return_relabeled:
        return best_misc, best_labeled_pred
    else:
        return best_misc
  
def cluster_purity(cm):
    ''' Compute the cluster purity index mentioned in Chen and He (2016)
    cm (2d-array): The confusion matrix resulting from the prediction
    --------------------------------------------------------------------
    returns (float): The cluster purity
    '''
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm) 

