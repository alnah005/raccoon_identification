# -*- coding: utf-8 -*-
"""
file: clustering.py

@author: Suhail.Alnahari

@description: 

@created: 2021-04-08T17:50:09.624Z-05:00

@last-modified: 2021-04-12T11:00:31.323Z-05:00
"""

# standard library
# 3rd party packages
# local source

import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import silhouette_score
import torch
from scipy.sparse import csr_matrix
from tqdm import tqdm
import pandas as pd
from scipy import stats
import random

def to_one_hot(y, n_dims=None):
    """ Take integer tensor with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims)
    for i in range(y_tensor.size()[0]):
        y_one_hot[i,y.view(-1,)[i]] = 1
    return y_one_hot

X = torch.load("test_imgs_FashionMNIST_64_100classes.pt")[:1000]

Graph = torch.load("test_label_FashionMNIST_64_100classes.pt")[:1000]

one_hot = to_one_hot(Graph)

labels_connection = one_hot@one_hot.T
for i in range(labels_connection.shape[0]):
    labels_connection[i,i] = 0
    
labels_connection = csr_matrix(labels_connection)


# from sklearn.manifold import TSNE
# tsne_model = TSNE(n_components=2, random_state=0,n_iter=5000,n_iter_without_progress=500,perplexity=35)
# tsne = tsne_model.fit_transform(X.cpu().detach().numpy())
tsne = torch.load("tsne_FashionMNIST_64_100classes.pt")[:1000]

# Create a graph capturing local connectivity. Larger number of neighbors
# will give more homogeneous clusters to the cost of computation
# time. A very large number of neighbors gives more evenly distributed
# cluster sizes, but may not impose the local manifold structure of
# the data
# knn_graph = kneighbors_graph(X, 30, include_self=False)
conn = ["none","labels"]
sil: Dict[str,Dict[int,Dict[str,float]]] = {}
for conn_index, connectivity in tqdm(enumerate((None, labels_connection)),desc="Connectivity\n\n\n"):
    sil[conn[conn_index]] = {}
    for n_clusters in tqdm((3,5,7,8,9,10,11,12,15,17,20,21,23,24,25,26,27,28,29,30),desc=f"\t {conn[conn_index]} num_cluster\n\n"):
        sil[conn[conn_index]][n_clusters] = {}
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
            sil[conn[conn_index]][n_clusters][linkage] = silhouette_score(X, model.labels_, metric = 'euclidean')
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

results = {i:None for i in sil.keys()}
for i in sil.keys():
    result = []
    for j in sil[i].keys():
        cluster_size = [j]
        for k in sil[i][j].keys():
            cluster_size.append(sil[i][j][k])
        result.append(cluster_size+[np.average(cluster_size[1:])])
    results[i] = pd.DataFrame(np.asarray(result),columns=["cluster_size"]+list(sil[i][j].keys())+['coeff_avg'])
    results[i].to_csv(f"{i}.csv",index=False)

optimal_cluster_sizes = [int(results[i]['cluster_size'][results[i]['coeff_avg'].argmax()]) for i in sil.keys()]


X_train = torch.load("train_imgs_FashionMNIST_64_100classes.pt")[:1000]

clus = random.choice(optimal_cluster_sizes)
model = AgglomerativeClustering(
    linkage='ward',
    connectivity=None,
    n_clusters=clus)
model.fit(torch.cat((X_train,X)))
new_train_labels = model.labels_

torch.save(torch.tensor(new_train_labels[:len(X_train)]),f"train_labels_FashionMNIST_64_100classes_to_{clus}classes.pt")
torch.save(torch.tensor(new_train_labels[len(X_train):]),f"test_labels_FashionMNIST_64_100classes_to_{clus}classes.pt")



