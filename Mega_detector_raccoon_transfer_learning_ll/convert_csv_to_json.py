# -*- coding: utf-8 -*-
"""
file: convert_csv_to_json.py

@author: Suhail.Alnahari

@description: File used to convert csv bounding box labels to JSON in the form of the JSON output from Microsoft's Mega detector

@created: 2021-02-15T10:40:00.541Z-06:00

@last-modified: 2021-03-02T15:45:40.168Z-06:00
"""

# standard library
# 3rd party packages
# local source

import pandas as pd
from dataclasses import dataclass
import json
import os

classLabel = {
    'Raccoon':1,
    'Skunk':2,
    'Cat':3,
    'Human':4,
    'Fox':5,
    'Other':6
}
labels = pd.read_csv('university-of-wyoming-raccoon-project-aggregated-for-retinanet.csv',header=None)

print(labels.keys())

labels = labels.values.tolist()

dicName: dict = {i[0].split('/')[-1]:[] for i in labels}
for i in labels:
    dicName[i[0].split('/')[-1]].append(i)

dic: dict = {'images':[]}

listOfFiles = os.listdir('videoImages')
height = 1080
width = 1920

for fileName in dicName:
    if (fileName in listOfFiles):
        res = {'detections':[],'file':"/content/drive/MyDrive/Zooniverse/videoImages/"+fileName,'max_detection_conf':1.00}
        for k in dicName[fileName]:
            xmins = k[1] / width
            xmaxs = k[3] / width
            ymins = k[2] / height
            ymaxs = k[4] / height
            detection = {'bbox': [xmins,ymins,abs(xmaxs-xmins),abs(ymaxs-ymins)],'category':str(classLabel[k[-1]]),'conf':1.00}
            res['detections'].append(detection)
        dic['images'].append(res)
dic["detection_categories"] = {
    "1": "raccoon",
    "2": "skunk",
    "3": "cat",
    "4": "human",
    "5": "fox",
    "6": "other"}
 
dic["info"] = {
     "detection_completion_time": "2021-03-02 20:01:22",
     "format_version": "1.0"
     }
 
with open('raccoon_true.json', 'w') as outfile:
    json.dump(dic, outfile,indent=1)