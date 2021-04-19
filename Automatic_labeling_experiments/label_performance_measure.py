# -*- coding: utf-8 -*-
"""
file: label_performance_measure.py

@author: Suhail.Alnahari

@description: 

@created: 2021-04-16T10:35:33.680Z-05:00

@last-modified: 2021-04-17T13:08:21.732Z-05:00
"""

# standard library
# 3rd party packages
# local source

import torch
from typing import List, Tuple, Any
import random 
# labels1 = torch.load("train_real.pt")
# labels2 = torch.load("train_prediction.pt")

# labels1 = torch.tensor([1,1,1,1,2,2,2,3,3,4,4,4,4,5])
# labels2 = torch.tensor([2,2,3,1,1,2,1,3,4,1,1,2,3,1])

# labels1 = torch.tensor([1,1,1,2,2,2,3,3,4,2])
# labels2 = torch.tensor([2,3,1,1,1,3,2,2,1,3])


def match_label_with_predictions(chosen_class,eliminated_classes,error,labels1,labels2):
    true_class_samples = labels2[labels1==chosen_class]
    for eliminated_class in eliminated_classes:
        true_class_samples = true_class_samples[true_class_samples != eliminated_class]
    possible_class_labels = torch.unique(true_class_samples)
    if (len(possible_class_labels) <= 0):
        return eliminated_classes,error+len(labels2[labels1==chosen_class])
    possible_class_counts = {class_number.item(): len(labels2[labels2==class_number]) for class_number in possible_class_labels} 
    within_class_counts = {class_number.item(): len(true_class_samples[true_class_samples==class_number]) for class_number in possible_class_labels}
    class_count_difference = {class_number.item(): possible_class_counts[class_number.item()]-within_class_counts[class_number.item()]  for class_number in possible_class_labels}
    maximum1 = max(list(class_count_difference.values()))
    possible_sorted_classes = sorted([(class_name,within_class_counts[class_name]) for class_name, count in class_count_difference.items() if count >= maximum1],reverse=True)
    maximum2 = max([count for _,count in possible_sorted_classes])
    possible_sorted_classes = [class_name for class_name,count in possible_sorted_classes if count >= maximum2]
    possible_class:Tuple[Any,int] = possible_sorted_classes[-1]
    eliminated_classes.append(possible_class)
    assert len(list(set(eliminated_classes))) == len(eliminated_classes)
    error += len(labels2[labels1==chosen_class]) - within_class_counts[possible_class]
    return eliminated_classes, error

def calculate_error(labels1,labels2):
    assert labels1.shape == labels2.shape
    assert len(labels1.shape) == 1
    class_labels = torch.unique(labels1)
    for k in class_labels:
        assert int == type(k.item())
    class_labels2 = torch.unique(labels2)
    for k in class_labels2:
        assert int == type(k.item())
        
    counts = {class_number.item(): len(labels1[labels1==class_number]) for class_number in class_labels} 
    eliminated_classes: List[Tuple[Any,int]] = []
    error = 0
    sampling_length = 100
    while(len(counts) > 0):
        maximum = max(list(counts.values()))
        possible_predicted_classes = [class_name for class_name, count in counts.items() if count >= maximum]
        sampling_errors_min = len(labels1)
        eliminated_classes_min = [*eliminated_classes]
        for k in range(sampling_length):
            sorted_classes = [*possible_predicted_classes]
            random.shuffle(sorted_classes)
            current_error = 0
            current_eliminated_classes = [*eliminated_classes]
            for i in sorted_classes:
                current_eliminated_classes, current_error = match_label_with_predictions(i,current_eliminated_classes,current_error,labels1,labels2)
            if (current_error < sampling_errors_min):
                sampling_errors_min = current_error
                eliminated_classes_min = current_eliminated_classes
        eliminated_classes = eliminated_classes_min
        error += sampling_errors_min
        counts = {class_name: class_count for class_name, class_count in counts.items() if class_name not in sorted_classes}
    return error/len(labels1)

# calculate_error(labels1,labels2)