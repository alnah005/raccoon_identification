# -*- coding: utf-8 -*-
"""
file: config.py

@author: Suhail.Alnahari

@description: This file contains all possible configurations for the [metric_loss.py] file

@created: 2021-04-07T09:33:39.899Z-05:00

@last-modified: 2021-04-07T09:59:28.358Z-05:00
"""

# standard library
# 3rd party packages
from torchvision import transforms, models
import torch.nn as nn
import torch
from pytorch_metric_learning.utils import common_functions
from pytorch_metric_learning import losses, miners, distances, reducers, samplers, trainers, testers

# local source
import local_dataset as DS


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Set trunk model and replace the softmax layer with an identity function
trunk = models.resnet50(pretrained=True)
trunk_output_size = trunk.fc.in_features
trunk.fc = common_functions.Identity()
trunk = torch.nn.DataParallel(trunk.to(device))

# Set embedder model. This takes in the output of the trunk and outputs 64 dimensional embeddings
embedder = torch.nn.DataParallel(MLP([trunk_output_size, 64],trunk).to(device))


# Set optimizers
embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=0.001, weight_decay=0.0001)
trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=0.0001, weight_decay=0.0001)

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

feedback_every = 10
def feedback_callback(epoch, i, loss, miner) -> str:
    return f"Epoch {epoch} Iteration {i}: Loss = {loss}, Number of mined triplets = {miner.num_triplets}"
