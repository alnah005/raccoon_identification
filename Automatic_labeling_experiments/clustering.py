# -*- coding: utf-8 -*-
"""
file: clustering.py

@author: Suhail.Alnahari

@description: 

@created: 2021-04-08T17:50:09.624Z-05:00

@last-modified: 2021-04-09T11:20:34.826Z-05:00
"""

# standard library
# 3rd party packages
# local source

import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import torch
from scipy.sparse import csr_matrix
from tqdm import tqdm

def to_one_hot(y, n_dims=None):
    """ Take integer tensor with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims)
    for i in range(y_tensor.size()[0]):
        y_one_hot[i,y.view(-1,)[i]] = 1
    return y_one_hot

X = torch.load("train_imgs_FashionMNIST_64_30classes.pt")

Graph = torch.load("train_label_FashionMNIST_64_30classes.pt")

one_hot = to_one_hot(Graph)

labels_connection = one_hot@one_hot.T
for i in range(labels_connection.shape[0]):
    labels_connection[i,i] = 0
    
labels_connection = csr_matrix(labels_connection)


# from sklearn.manifold import TSNE
# tsne_model = TSNE(n_components=2, random_state=0,n_iter=5000,n_iter_without_progress=500,perplexity=35)
# tsne = tsne_model.fit_transform(X.cpu().detach().numpy())
tsne = torch.load("tsne_FashionMNIST_64_30classes.pt")

# Create a graph capturing local connectivity. Larger number of neighbors
# will give more homogeneous clusters to the cost of computation
# time. A very large number of neighbors gives more evenly distributed
# cluster sizes, but may not impose the local manifold structure of
# the data
# knn_graph = kneighbors_graph(X, 30, include_self=False)
conn = ["none","labels"]

for conn_index, connectivity in tqdm(enumerate((None, labels_connection)),desc="Connectivity\n\n\n"):
    for n_clusters in tqdm((3,5,10,20,30,40,50),desc=f"\t {conn[conn_index]} num_cluster\n\n"):
        plt.figure(figsize=(10, 4))
        for index, linkage in tqdm(enumerate(('average',
                                         'complete',
                                         'ward',
                                         'single')),desc=f"\t\t {n_clusters} linkage \n"):
            plt.subplot(1, 4, index + 1)
            model = AgglomerativeClustering(linkage=linkage,
                                            connectivity=connectivity,
                                            n_clusters=n_clusters)
            t0 = time.time()
            model.fit(X)
            elapsed_time = time.time() - t0
            plt.scatter(tsne[:, 0], tsne[:, 1], c=model.labels_,
                        cmap=plt.cm.nipy_spectral)
            plt.title('linkage=%s\n(time %.2fs)' % (linkage, elapsed_time),
                      fontdict=dict(verticalalignment='top'))
            plt.axis('equal')
            plt.axis('off')

            plt.subplots_adjust(bottom=0, top=.83, wspace=0,
                                left=0, right=1)
            plt.suptitle('n_cluster=%i, connectivity=%r' %
                         (n_clusters, connectivity is not None), size=17)
        plt.savefig(f"{conn[conn_index]}_{n_clusters}_{index}_{linkage}_64_FashionMNIST.png")