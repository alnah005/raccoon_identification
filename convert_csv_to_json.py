# -*- coding: utf-8 -*-
"""
file: convert_csv_to_json.py

@author: Suhail.Alnahari

@description: File used to convert csv bounding box labels to JSON in the form of the JSON output from Microsoft's Mega detector

@created: 2021-02-15T10:40:00.541Z-06:00

@last-modified: 2021-02-19T14:00:39.384Z-06:00
"""

# standard library
# 3rd party packages
# local source

import pandas as pd
from dataclasses import dataclass
import json

classLabel = {
    'Raccoon':0,
    'Skunk':1,
    'Cat':2,
    'Human':3,
    'Fox':4,
    'Other':5
}
labels = pd.read_csv('university-of-wyoming-raccoon-project-aggregated-for-retinanet.csv',header=None)

print(labels.keys())

labels = labels.values.tolist()

dicName: dict = {i[0].split('/')[-1]:[] for i in labels}
for i in labels:
    dicName[i[0].split('/')[-1]].append(i)

dic: dict = {'images':[]}

for fileName in dicName:
    res = {'detections':[],'file':fileName,'max_detection_conf':1.00}
    for k in dicName[fileName]:
        detection = {'bbox': k[1:-1],'category':classLabel[k[-1]],'conf':1.00}
        res['detections'].append(detection)
    dic['images'].append(res)

with open('raccoon_true.json', 'w') as outfile:
    json.dump(dic, outfile,indent=1)