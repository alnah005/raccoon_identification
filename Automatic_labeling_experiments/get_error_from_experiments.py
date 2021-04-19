# -*- coding: utf-8 -*-
"""
file: get_error_from_experiments.py

@author: Suhail.Alnahari

@description: 

@created: 2021-04-16T14:19:12.828Z-05:00

@last-modified: 2021-04-17T13:08:13.615Z-05:00
"""

# standard library
import os

# 3rd party packages
import torch

# local source
import label_performance_measure as lpm

base_dir = './joblib'
experiments_dirs = [direc for direc in os.listdir(base_dir) if 'experiment' == direc[:len('experiment')]]
numbers = sorted([int(i.split('_')[1]) for i in experiments_dirs])

_, labels1_train = torch.load("./processed/training.pt")
_, labels1_test = torch.load("./processed/test.pt")
labels1 = torch.cat((labels1_train,labels1_test))
final_errors = []
comparing_accross_time: lpm.List[lpm.List[int]] = []
labels_size = []
labels_before = None
for i in numbers:
    cwd = os.path.join(base_dir,f"experiment_{i}")
    pred_train_labels = torch.load(os.path.join(cwd,'train_clustering_labels.pt'))
    pred_test_labels = torch.load(os.path.join(cwd,'test_clustering_labels.pt'))
    labels2 = torch.cat((pred_train_labels,pred_test_labels))
    labels_size.append(len(torch.unique(labels2)))
    final_errors.append(lpm.calculate_error(labels1,labels2))
    if not(labels_before is None):
        comparing_accross_time.append(lpm.calculate_error(labels_before,labels2))
    labels_before = labels2.detach()
        
print("\n".join([str(elem) for elem in final_errors]))

import matplotlib.pyplot as plt
import numpy as np

plt.figure()
plt.plot(np.arange(len(final_errors)),np.asarray(labels_size))
plt.show()

plt.figure()
plt.plot(np.arange(len(final_errors)),np.asarray(final_errors),label='Isomerphic Sim')
plt.legend()
plt.show()

print("\n".join([str(elem) for elem in comparing_accross_time]))

plt.figure()
plt.plot(np.arange(len(comparing_accross_time)),np.asarray(comparing_accross_time),label='Isomerphic Sim')
plt.legend()
plt.show()

# lpm.calculate_error(torch.tensor([1,2,3,3,4,5,6]),torch.tensor([1,2,3,3,4,5,6]))
