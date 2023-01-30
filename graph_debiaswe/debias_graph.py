# -*- coding: utf-8 -*-
# @Author: Ashutosh Tiwari
# @Email: checkashu@gmail.com
# @Date:   2023-01-30 12:07:40
# @Filepath: graph_debiaswe/debias_graph.py

# assumptions for converting to graph
# bias direction: taken top 1 component of PCA or LDA   
# defitional pairs: pairs of centroids of groups
# 
import numpy as np
from utils import get_direction


def debias_wrapper(embs, gender_specific_words, definitional, equalize, direction_method='PCA', y=None):
    nodes, dim = embs.shape
    assert gender_specific_words.shape[1] == dim
    assert definitional.shape == 
    direction = get_direction(embs, y, direction_method)
    
    
def debias_graph(embs: np.array, gender_direction: np.array, definitional: np.array, equalizer: np.array, y:np.array=None):
    """debias graph embeddings

    Args:
        embs (np.array): embeddings to be debiased
        gender_direction (np.array): direction to be debiased
        definitional (_type_): defitional pairs, may be centroid of groups
        equalizer (_type_):  equalizer pairs, may be the most distant pairs
        y (np.array, optional): node labels, for LDA
    """
    assert gender_direction.shape[0] == embs.shape[1], "gender_direction and embs should have same dimension"
    