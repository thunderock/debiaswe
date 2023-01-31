# -*- coding: utf-8 -*-
# @Author: Ashutosh Tiwari
# @Email: checkashu@gmail.com
# @Date:   2023-01-30 12:07:40
# @Filepath: graph_debiaswe/debias_graph.py

# assumptions for converting to graph
# gender specific words: node ids of group specific words, probably the one closest to centroids
# defitional pairs: pairs of centroids of groups
# equalize words: these should be equidistant from the centroids of the groups, so these are the most distant pairs

import numpy as np
from utils import get_direction, EMB_UTILS
import we

def debias_wrapper(embs, gender_specific_words, definitional, equalize, direction_method='PCA', y=None):
    nodes, dim = embs.shape
    direction = get_direction(embs, y, direction_method)
    # gender specific words are the node ids, lets have a vector of size 1x nodes
    # where i == true denotes that it is gender specific
    # definitional are not the node ids, but the centroids of the groups
    # equalize are pairs of node ids
    assert definitional.shape[1:] == (dim, 2)
    assert direction.shape == (dim, )
    assert equalize.shape[1:] == (2, )


    for i in range(nodes):
        if not gender_specific_words[i]:
            embs[i] = we.drop(embs[i], direction)

    EMB_UTILS.normalize(embs)

    for (a,b) in definitional:

        y = we.drop(embs[a] + embs[b] / 2, direction)
        z = np.sqrt(1 - np.linalg.norm(y)**2)
        if (embs[a] - embs[b]).dot(direction) < 0:
            z = -z
        embs[a] = y + z * direction
        embs[b] = y - z * direction

    EMB_UTILS.normalize(embs)

    return embs

    
