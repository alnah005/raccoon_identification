# -*- coding: utf-8 -*-
"""
file: dataset.py

@author: Suhail.Alnahari

@description: 

@created: 2021-04-06T18:27:51.082Z-05:00

@last-modified: 2021-04-06T19:20:09.208Z-05:00
"""

# standard library
# 3rd party packages
# local source
import os
import pandas as pd
from PIL import Image

class RaccoonDataset(object):
    labels =None
    def __init__(self,
                 img_folder = "../Generate_Individual_IDs_dataset/croppedImages",
                 labels = "labels.csv",
                 transforms = None
                 ):
        self.img_folder = img_folder
        self.labels_path = os.path.join(img_folder,labels)
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted([i for i in os.listdir(self.img_folder) if (('.png' in i) or ('.jpg' in i) or ('.jpeg' in i))]))
        self.transforms = transforms

    def refresh(self):
        self.imgs = list(sorted(os.listdir(os.path.join(self.root, self.img_folder))))

    def __getitem__(self, idx):
        # load images ad masks
        if(isinstance(idx, str)):
            try:
                idx = int(idx)
            except:
                try:
                    if ((idx[-4:] == '.png') or (idx[-4:] == '.jpg')):
                        idx = self.imgs.index(idx[:-4]+'.jpg')
                    else:
                        idx = self.imgs.index(idx)
                except:
                    print("invalid index")
                    idx = np.random.randint(low=0,high=len(self.imgs))
        img_path = os.path.join(self.img_folder, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        if not(self.transforms is None):
            img = self.transforms(img)
        if (self.labels is None):
            fle = open(self.labels_path)
            res = {}
            for i in fle:
                res[i.split(',')[0]] = i.split(',')[1]
            fle.close()
            self.labels = res
        return img, int(self.labels[self.imgs[idx]].replace('\n','').replace(' ','').replace(',',''))


    def __len__(self):
        return len(self.imgs)
