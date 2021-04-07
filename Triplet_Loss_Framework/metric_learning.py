# -*- coding: utf-8 -*-
"""
file: metric_learning.py

@author: Suhail.Alnahari

@description: Metric learning training file that runs based on [config.py] settings

@created: 2021-04-05T11:18:24.742Z-05:00

@last-modified: 2021-04-07T10:01:27.953Z-05:00
"""

# standard library
import logging

# 3rd party packages
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import numpy as np
import pytorch_metric_learning

# local source
from config_local import (
    train_transform, val_transform, device,
    torch, embedder, embedder_optimizer,
    trunk_optimizer, train_dataset, batch_size,num_epochs,
    train_loader, miner, metric_loss, val_dataset,
    feedback_every, feedback_callback
    )

logging.getLogger().setLevel(logging.INFO)
logging.info("VERSION %s"%pytorch_metric_learning.__version__)



def get_all_embeddings(dataset, model, data_device):
    # dataloader_num_workers has to be 0 to avoid pid error
    # This only happens when within multiprocessing
    tester = testers.BaseTester(dataloader_num_workers=0, data_device=data_device)
    return tester.get_all_embeddings(dataset, model)

### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test_implem(train_set, test_set, model, accuracy_calculator, data_device):
    train_embeddings, train_labels = get_all_embeddings(train_set, model, data_device)
    test_embeddings, test_labels = get_all_embeddings(test_set, model, data_device)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(test_embeddings, 
                                                train_embeddings,
                                                test_labels,
                                                train_labels,
                                                False)
    print("Validation set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))
    print("Validation set accuracies blob:\n {}".format(accuracies))

def test_model(train_set, test_set, model, epoch, data_device):
    print("Computing validation set accuracy for epoch {}".format(epoch))
    accuracy_calculator = AccuracyCalculator(include = ("precision_at_1",),avg_of_avgs=True, k = 1)
    test_implem(train_set, test_set, model, accuracy_calculator, data_device)


for epoch in range(num_epochs):
    epoch_loss = 0.
    print("Starting epoch {}".format(epoch))
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        embedder_optimizer.zero_grad()
        trunk_optimizer.zero_grad()
        output = embedder(data)
        hard_pairs = miner(output, target)
        loss = metric_loss(output, target, hard_pairs)
        epoch_loss += loss.item()
        loss.backward()
        embedder_optimizer.step()
        trunk_optimizer.step()
        if i % feedback_every == 0:
            print(feedback_callback(epoch, i, loss, miner.num_triplets))
    print('Epoch {}, average loss {}'.format(epoch, epoch_loss/len(train_loader)))
    test_model(train_dataset, val_dataset, embedder, epoch, device)
