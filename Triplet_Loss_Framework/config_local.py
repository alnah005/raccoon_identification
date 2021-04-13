# -*- coding: utf-8 -*-
"""
file: config.py

@author: Suhail.Alnahari

@description: This file contains all possible configurations for the [metric_loss.py] file

@created: 2021-04-07T09:33:39.899Z-05:00

@last-modified: 2021-04-13T14:54:36.968Z-05:00
"""

# standard library
import os

# 3rd party packages
from torchvision import transforms, models, datasets
import torch.nn as nn
import torch
from pytorch_metric_learning.utils import common_functions
from pytorch_metric_learning import losses, miners, distances, reducers

# local source
import local_dataset as DS


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
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
        return self.net(x)

class Embedder(nn.Module):
    def __init__(self, trunk,embedder_head, trunk_optimizer,embedder_head_optimizer,checkpointLocation="./experiment"):
        super().__init__()
        self.trunk = trunk
        self.embedder_head = embedder_head
        self.trunk_optimizer = trunk_optimizer
        self.embedder_head_optimizer = embedder_head_optimizer
        self.checkpointLocation = checkpointLocation
        if not(os.path.isdir(checkpointLocation)):
            os.mkdir(checkpointLocation)

    def __call__(self,x):
        return self.forward(x)
    
    def train(self):
        self.embedder_head.train()
        self.trunk.train()

    def eval(self):
        self.embedder_head.eval()
        self.trunk.eval()

    def optimize(self):
        self.embedder_head_optimizer.step()
        self.trunk_optimizer.step()

    def zero_grad(self):
        self.trunk_optimizer.zero_grad()
        self.embedder_head_optimizer.zero_grad()

    def forward(self, x):
        return self.embedder_head(self.trunk(x))

    def save(self, epoch):
        torch.save({
            'epoch': epoch,
            'trunk_state_dict': self.trunk.state_dict(),
            'trunk_optimizer_state_dict': self.trunk_optimizer.state_dict(),
            'embedder_head_state_dict': self.embedder_head.state_dict(),
            'embedder_head_optimizer_state_dict': self.embedder_head_optimizer.state_dict(),
            }, os.path.join(self.checkpointLocation,f"model_{epoch}.pt"))
    
    def load(self):
        path = self._findMostRecent()
        if (len(path) == 0):
            return None
        checkpoint = torch.load(os.path.join(self.checkpointLocation,path))
        self.trunk.load_state_dict(checkpoint['trunk_state_dict'])
        self.trunk_optimizer.load_state_dict(checkpoint['trunk_optimizer_state_dict'])
        self.embedder_head.load_state_dict(checkpoint['embedder_head_state_dict'])
        self.embedder_head_optimizer.load_state_dict(checkpoint['embedder_head_optimizer_state_dict'])
        epoch = checkpoint['epoch']
        self.train()
        return epoch
    
    def _findMostRecent(self) -> str:
        model_ckps = list(sorted([i for i in os.listdir(self.checkpointLocation) if ('.pt' in i)]))
        if (len(model_ckps) == 0):
            return ""
        latest_model = ""
        maximum_epoch = 0
        for i in model_ckps:
            _, epoch = i.split('_')
            epoch = int(epoch[:-3])
            if (epoch > maximum_epoch):
                maximum_epoch = epoch
                latest_model = i
        print(f"Loading model {latest_model} at epoch {maximum_epoch}")
        return latest_model

# Set the image transforms
train_transform = transforms.Compose([
                                    # transforms.Resize((250,200)),
                                    # transforms.RandomResizedCrop(scale=(0.16, 1), ratio=(0.75, 1.33), size=64),
                                    transforms.Lambda(lambda image: image.convert('RGB')),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])

val_transform = transforms.Compose([
                                    # transforms.Resize((250,200)),
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
embedder_head = torch.nn.DataParallel(MLP([trunk_output_size, 64]).to(device))


# Set optimizers
embedder_head_optimizer = torch.optim.Adam(embedder_head.parameters(), lr=0.001, weight_decay=0.0001)
trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=0.0001, weight_decay=0.0001)

def read_checkpoint_config(ckpt_loc="/home/fortson/alnah005/raccoon_identification/Triplet_Loss_Framework/experiment",config_name='config.txt'):
    f = open(os.path.join(ckpt_loc,config_name))
    lines = f.readlines()
    f.close()
    if (len(lines) > 0):
        f = open(os.path.join(ckpt_loc,config_name),'a')
        f.write(f"experiment_{len(lines)}\n")
        f.close()
        return os.path.join(ckpt_loc,lines[-1].replace('\n','')),os.path.join(ckpt_loc,f"experiment_{len(lines)}")
    else:
        f = open(os.path.join(ckpt_loc,config_name),'a')
        f.write('experiment_0\n')
        f.close()
        return None, os.path.join(ckpt_loc,'experiment_0')

prev_checkpoint,checkpoint_loc = read_checkpoint_config()

embedder = Embedder(trunk,embedder_head,trunk_optimizer,embedder_head_optimizer,checkpointLocation=checkpoint_loc)

# Set Metric learning parameters
distance = distances.LpDistance(normalize_embeddings=True,p=2,power=1)
reducer_dict = {"triplet": reducers.ThresholdReducer(0.1), "triplet": reducers.MeanReducer()}
reducer = reducers.MultipleReducers(reducer_dict)
metric_loss = losses.TripletMarginLoss(margin = 0.2, distance = distance, reducer = reducer,swap=True)
miner = miners.TripletMarginMiner(margin = 0.2, distance = distance, type_of_triplets = "semihard")


# Set other training parameters
batch_size = 64
num_epochs = 20

class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self,dataset,train=True,split_targets=False):
        self.dataset = dataset
        self.targets = None
        self.split_targets = split_targets
        if not(prev_checkpoint is None):
            if (train):
                print("loading from "+os.path.join(prev_checkpoint,'train_clustering_labels.pt'))
                self.targets = torch.load(os.path.join(prev_checkpoint,'train_clustering_labels.pt'))
            else:
                print("loading from "+os.path.join(prev_checkpoint,'test_clustering_labels.pt'))
                self.targets = torch.load(os.path.join(prev_checkpoint,'test_clustering_labels.pt'))
    
    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        if (self.split_targets):
            target += (idx%10)*10
        if not(self.targets is None):
            target = self.targets[idx].item()
        return img, target
    
    def __len__(self):
        return len(self.dataset)

    def refresh(self):
        try:
            self.dataset.refresh()
        except:
            print("no refresh method found")     
    def __iter__(self,idx=0):
        try:
            while(idx < len(self)):
                idx+= 1
                yield self[idx-1]
        except:
            print("no iter method found")     

# train_dataset = DatasetWrapper(DS.RaccoonDataset(img_folder="/home/fortson/alnah005/raccoon_identification/Generate_Individual_IDs_dataset/croppedImages/train",transforms = train_transform))
# val_dataset = DatasetWrapper(DS.RaccoonDataset(img_folder="/home/fortson/alnah005/raccoon_identification/Generate_Individual_IDs_dataset/croppedImages/test", transforms = val_transform),train=False)

train_dataset = DatasetWrapper(datasets.FashionMNIST(root = './', train=True, download=True, transform=train_transform),split_targets=True)
val_dataset = DatasetWrapper(datasets.FashionMNIST(root = './', train=False, download=True, transform=val_transform),train=False,split_targets=True)


train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=1)
test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,num_workers=1)

feedback_every = 3
def feedback_callback(epoch, i, loss, miner) -> str:
    return f"Epoch {epoch} Iteration {i}: Loss = {loss}, Number of mined triplets = {miner.num_triplets}"

save_model_every_epochs = 3
