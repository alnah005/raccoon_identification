# -*- coding: utf-8 -*-
"""
file: metric_learning.py

@author: Suhail.Alnahari

@description: Metric learning training file that runs based on [config.py] settings

@created: 2021-04-05T11:18:24.742Z-05:00

@last-modified: 2021-04-13T12:06:37.042Z-05:00
"""

# standard library
import logging
import os

# 3rd party packages
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
from pytorch_metric_learning import testers
import numpy as np
import pytorch_metric_learning

# local source
from config_local import (
    train_transform, val_transform, device,
    torch, embedder,train_dataset, batch_size,num_epochs,
    train_loader, miner, metric_loss, val_dataset,
    feedback_every, feedback_callback, save_model_every_epochs,
    checkpoint_loc
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

checkpoint_epoch = embedder.load()
if (checkpoint_epoch is None):
    print("Starting training from scratch")
    checkpoint_epoch = 0

for epoch in range(checkpoint_epoch,num_epochs):
    epoch_loss = 0.
    print("Starting epoch {}".format(epoch))
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        embedder.zero_grad()
        output = embedder(data)
        hard_pairs = miner(output, target)
        loss = metric_loss(output, target, hard_pairs)
        epoch_loss += loss.item()
        loss.backward()
        embedder.optimize()
        if i % feedback_every == 0:
            print(feedback_callback(epoch, i, loss, miner))
    if epoch % save_model_every_epochs ==0:
        embedder.save(epoch)
    print('Epoch {}, average loss {}'.format(epoch, epoch_loss/len(train_loader)))
    test_model(train_dataset, val_dataset, embedder, epoch, device)

result_train_img, result_train_label = get_all_embeddings(train_dataset,embedder,device)
result_test_img, result_test_label = get_all_embeddings(val_dataset,embedder,device)

torch.save(result_train_img.cpu(),os.path.join(checkpoint_loc,'train_imgs.pt'))
torch.save(result_test_img.cpu(),os.path.join(checkpoint_loc,'test_imgs.pt'))
torch.save(result_train_label.cpu(),os.path.join(checkpoint_loc,'train_label.pt'))
torch.save(result_test_label.cpu(),os.path.join(checkpoint_loc,'test_label.pt'))

from sklearn.manifold import TSNE
tsne_model = TSNE(n_components=2, random_state=0,n_iter=10000,n_iter_without_progress=500,perplexity=35)
embedding = [embedder.forward(sample[0].cuda()).detach() for sample in val_dataset]
tsne = tsne_model.fit_transform(torch.cat(embedding,dim=0).cpu().detach().numpy())
torch.save(torch.tensor(tsne),os.path.join(checkpoint_loc,'tsne_FashionMNIST_64_100classes.pt'))