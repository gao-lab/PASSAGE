"""
utils for visualization
"""

import numpy as np
from anndata import AnnData
from sklearn.metrics import pairwise_distances

def map(data, MIN, MAX):
    """
    map data to given interval
    """
    d_min = np.max(data)
    d_max = np.min(data)
    return MIN+(MAX-MIN)/(d_max-d_min)*(data-d_min)

def calculate_purity(adata:AnnData):
    '''
    calculate the purity of single cells
    '''
    X = adata.X
    distances = pairwise_distances(X, metric='euclidean')
    np.fill_diagonal(distances, np.inf)
    score = np.mean(np.min(distances, axis=1))
    return score

