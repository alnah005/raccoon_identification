# -*- coding: utf-8 -*-
"""
file: metric_learning.py

@author: Suhail.Alnahari

@description: 

@created: 2021-04-05T11:18:24.742Z-05:00

@last-modified: 2021-04-06T20:12:01.226Z-05:00
"""

# standard library
# 3rd party packages
# local source

from pytorch_metric_learning import losses, miners, distances, reducers, samplers, trainers, testers
from pytorch_metric_learning.utils import common_functions
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torch
import torch.nn as nn
from PIL import Image
import logging
import local_dataset as DS
import pytorch_metric_learning
logging.getLogger().setLevel(logging.INFO)
logging.info("VERSION %s"%pytorch_metric_learning.__version__)

class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, trunk, final_relu=False):
        super().__init__()
        layer_list = []
        self.trunk = trunk
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(self.trunk(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set trunk model and replace the softmax layer with an identity function
trunk = torchvision.models.resnet50(pretrained=True)
trunk_output_size = trunk.fc.in_features
trunk.fc = common_functions.Identity()
trunk = torch.nn.DataParallel(trunk.to(device))

# Set embedder model. This takes in the output of the trunk and outputs 64 dimensional embeddings
embedder = torch.nn.DataParallel(MLP([trunk_output_size, 64],trunk).to(device))


# Set optimizers
embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=0.001, weight_decay=0.0001)
trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=0.0001, weight_decay=0.0001)

# Set the image transforms
train_transform = transforms.Compose([
                                    transforms.Resize((250,200)),
                                    # transforms.RandomResizedCrop(scale=(0.16, 1), ratio=(0.75, 1.33), size=64),
                                    transforms.Lambda(lambda image: image.convert('RGB')),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

val_transform = transforms.Compose([
                                    transforms.Resize((250,200)),
                                    transforms.Lambda(lambda image: image.convert('RGB')),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

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
    print("Validation set accuracy (r_precision) = {}".format(accuracies["r_precision"]))
    print("Validation set accuracy (mean_average_precision_at_r) = {}".format(accuracies["mean_average_precision_at_r"]))

def test_model(train_set, test_set, model, epoch, data_device):
    print("Computing validation set accuracy for epoch {}".format(epoch))
    accuracy_calculator = AccuracyCalculator(include = ("precision_at_1","r_precision","mean_average_precision_at_r",),avg_of_avgs=True, k = 1)
    test_implem(train_set, test_set, model, accuracy_calculator, data_device)

distance = distances.LpDistance(normalize_embeddings=True,p=2,power=1)
reducer_dict = {"triplet": reducers.ThresholdReducer(0.1), "triplet": reducers.MeanReducer()}
reducer = reducers.MultipleReducers(reducer_dict)
metric_loss = losses.TripletMarginLoss(margin = 0.2, distance = distance, reducer = reducer,swap=True)
miner = miners.TripletMarginMiner(margin = 0.2, distance = distance, type_of_triplets = "semihard")


# Set other training parameters
batch_size = 64
num_epochs = 20
train_dataset = DS.RaccoonDataset(img_folder="/home/fortson/alnah005/raccoon_identification/Generate_Individual_IDs_dataset/croppedImages/train",transforms = train_transform)
val_dataset = DS.RaccoonDataset(img_folder="/home/fortson/alnah005/raccoon_identification/Generate_Individual_IDs_dataset/croppedImages/test", transforms = val_transform)

train_loader = torch.utils.data.DataLoader(train_dataset,pin_memory=True, batch_size=batch_size, shuffle=True,num_workers=1)
test_loader = torch.utils.data.DataLoader(val_dataset,pin_memory=True, batch_size=batch_size,num_workers=1)

num_batches = np.ceil(len(train_dataset) / batch_size)
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
        if i % batch_size == 0:
            print("Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(epoch, i, loss, miner.num_triplets))
    print('Epoch {}, average loss {}'.format(epoch, epoch_loss/len(train_loader)))
    test_model(train_dataset, val_dataset, embedder, epoch, device)
