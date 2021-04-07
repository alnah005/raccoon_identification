# -*- coding: utf-8 -*-
"""
file: get_images_from_frames_using_boxes.py

@author: Suhail.Alnahari

@description: 

@created: 2021-04-06T16:35:11.623Z-05:00

@last-modified: 2021-04-06T16:59:55.865Z-05:00
"""

# standard library
# 3rd party packages
# local source

import pandas as pd
import numpy as np
from PIL import Image
import os

imagesPath = "d:/Zoon_parent/Zooniverse/data/raccoons/images"
labelsPath = "d:/Zoon_parent/raccoon_identification/Generate_Individual_IDs_dataset/videoDataset.csv"
finalPath = "d:/Zoon_parent/raccoon_identification/Generate_Individual_IDs_dataset/croppedImages"
labels = pd.read_csv(labelsPath,header=None)
classLabels = {i:index for index,i in enumerate(np.unique(labels[6].values))}

f = open(os.path.join(finalPath,"labels.csv"),'w')
for index, i in enumerate(labels.values):
    name,x1,x2,y1,y2,label,vid = i
    image = Image.open(os.path.join(imagesPath,name))
    cropped = image.crop((x1,x2,y1,y2))
    cropped.save(os.path.join(finalPath,f'cropped_{index}_'+name))
    f.write(f'cropped_{index}_'+name+','+str(classLabels[vid])+'\n')
f.close()

    
    