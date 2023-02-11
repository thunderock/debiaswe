#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, sys
residual2vec_ = '../../residual2vec_'
sys.path.insert(0, residual2vec_)


# In[2]:


import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from debias_graph import debias_wrapper
from we import doPCA
from we_utils import get_direction
from utils import graph_utils
from models import fast_knn_cpu
from tqdm import tqdm


# In[3]:


BASE = "../../final_128/{}/"

def get_embs(dataset, run_no, deepwalk):
    y = pd.read_csv(BASE.format(dataset) + "node_table.csv").group_id.values
    deepwalk = np.load(BASE.format(dataset) + "{}_{}/{}_{}.npy".format(dataset, 
                                                                     run_no, dataset, 
                                                                       "deepwalk" if deepwalk else "node2vec"))
    
    
    centroids = graph_utils.get_centroid_per_group(deepwalk, y)
    # definitional words, these are supposed to be represent the group,
    # in this case lets take these to be the nodes closest to centroid of group
    # in this case are the centroids of the groups
    definitional = graph_utils.get_n_nearest_neighbors_for_nodes(
        nodes=centroids, 
        embs=deepwalk,
        k=1,
        metric='cosine'
    )
    
    N, dim = deepwalk.shape
    K = np.unique(y).shape[0]
    
    gender_specific_nodes = graph_utils.get_n_nearest_neighbors_for_nodes(
        nodes=centroids, 
        embs=deepwalk,
        k=int (.2 * N) // K,
        metric='cosine'
    )
    equalize = graph_utils.get_farthest_pairs(deepwalk, y, same_class=False, 
                                              per_class_count=1)
    
    direction = get_direction(deepwalk, y, "PCA")
    
    return debias_wrapper(emb=deepwalk, gender_specific_words=gender_specific_nodes, 
               definitional=None, equalize=equalize, y=y, direction=direction,
                          drop_gender_specific_words=True)
            
    
    


# In[4]:


datasets = ["polbook", "polblog", "airport", "pokec"]

runs = ["one", "two", "three", "four", "five"]

for dataset in datasets:
    for directory in tqdm(runs, desc=dataset + "_deepwalk"):
        embs = get_embs(dataset=dataset, run_no=directory, deepwalk=True)
        np.save(BASE.format(dataset) + "{}_{}/{}_baseline_man_woman+deepwalk.npy".format(
            dataset, directory, dataset),
               embs)
    
    for directory in tqdm(runs, desc=dataset + "_node2vec"):
        embs = get_embs(dataset=dataset, run_no=directory, deepwalk=False)
        np.save(BASE.format(dataset) + "{}_{}/{}_baseline_man_woman+node2vec.npy".format(
            dataset, directory, dataset),
               embs)
        


# In[5]:


# dataset = "pokec"
# run_no = "one"


# In[6]:


# y = pd.read_csv(BASE.format(dataset) + "node_table.csv").group_id.values
# deepwalk = np.load(BASE.format(dataset) + "{}_{}/{}_deepwalk.npy".format(dataset, 
#                                                                  run_no, dataset))
# centroids = graph_utils.get_centroid_per_group(deepwalk, y)
# definitional = graph_utils.get_n_nearest_neighbors_for_nodes(
#         nodes=centroids, 
#         embs=deepwalk,
#         k=1,
#         metric='cosine'
#     )
    
# dim = deepwalk.shape[1]
# N = np.unique(y).shape[0]
# gender_specific_nodes = graph_utils.get_n_nearest_neighbors_for_nodes(
#     nodes=centroids, 
#     embs=deepwalk,
#     k=10,
#     metric='cosine'
# )
# equalize = graph_utils.get_farthest_pairs(deepwalk, y, same_class=False, 
#                                               per_class_count=1)

