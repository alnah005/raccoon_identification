# -*- coding: utf-8 -*-
"""
file: box_compare.py

@author: Suhail.Alnahari

@description: File used to analyze the differences between Volunteer labels and Microsoft's Mega Detector

@created: 2021-02-15T10:20:13.627Z-06:00

@last-modified: 2021-02-19T13:58:55.824Z-06:00
"""

# standard library
# 3rd party packages
# local source

import json
from dataclasses import dataclass

def convert_y1_x1_y2_x2(tf_coords):
    return [tf_coords[0]*1920, tf_coords[1]*1080, (tf_coords[0]+tf_coords[2])*1920,(tf_coords[1]+tf_coords[3])*1080]
    
def sorted_enumerate(seq):
    args = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(seq))]
    values = sorted(seq)
    result = []
    for i in range(len(seq)):
        result.append((args[i],values[i]))
    return result

@dataclass
class Detection:
    def __init__(self, detection,convert):
        self.bbox = detection['bbox']
        if (convert):
            self.bbox = convert_y1_x1_y2_x2(self.bbox)
        self.category = detection['category']
        self.conf = detection['conf']
    
    def __post_init__(self):
        assert isinstance(self.bbox, list)
        assert isinstance(self.conf, float)


@dataclass
class ImageDetection:
    def __init__(self,detections,name,confidence,convertBox=False):
        self.detections = [Detection(i,convertBox) for i in detections]
        self.fileName = name.split('/')[-1]
        self.maxConfidence = confidence
    def __post_init__(self):
        assert isinstance(self.detections, list)
        assert isinstance(self.fileName, str)
        assert 'jpg' in self.fileName


with open('raccoon.json') as json_file:
    data = json.load(json_file)

with open('raccoon_true.json') as json_file:
    data2 = json.load(json_file)

    
imagePreds = [ImageDetection(i['detections'],i['file'],i['max_detection_conf'],convertBox=True) for i in data['images']]

imageDict = {i.fileName: i for i in imagePreds}

imagePreds2 = [ImageDetection(i['detections'],i['file'],i['max_detection_conf']) for i in data2['images']]

imageDict2 = {i.fileName: i for i in imagePreds2}


def compare_numbers(detections1,detections2):
    return abs(len(detections1)-len(detections2))

def Euclidean(point1,point2):
    assert(len(point1)==2)
    assert(len(point2)==2)
    return (abs(point2[1]-point1[1])**2 + abs(point2[0]-point1[0])**2)**0.5

def coordsToCenter(dets):
    return [((d.bbox[2]+d.bbox[0])/2, (d.bbox[3]+d.bbox[1])/2) for d in dets]


def removeCenter(center,c_prefs):
    return [[(a[0],a[1]) for a in k if a[0] != center] for k in c_prefs]

def matchClosestCenters(c_prefs):
    selected = [0 for i in range(len(c_prefs))]
    center1 = 0
    center2 = 0
    distance = 100000000000
    result = []
    while sum(selected) < len(c_prefs):
        for i in range(len(selected)):
            if (selected[i] == 0):
                if (len(c_prefs[i]) > 0):
                    if (distance > c_prefs[i][0][1]):
                        center1 = i
                        center2 = c_prefs[i][0][0]
                        distance = c_prefs[i][0][1]
                else:
                    selected[i] = 1
                    result.append((i,None))
        if (selected[center1]==0):
            result.append((center1,center2))
            distance = 100000000000
            selected[center1] = 1
            c_prefs = removeCenter(center2,c_prefs)

    return result




def getCenterPairs(det1,det2):
    centers1 = coordsToCenter(det1)
    centers2 = coordsToCenter(det2)
    rev = False
    if (len(centers1) < len(centers2)):
        temp = centers1
        centers1 = [i for i in centers2]
        centers2 = temp
        rev = True

    center_pref_distance = [None for i in range(len(centers1))]

    for index,i in enumerate(centers1):
        distances = [Euclidean(i,j) for j in centers2]
        center_pref_distance[index] = sorted_enumerate(distances)

    centerPairsIndexes = matchClosestCenters(center_pref_distance)
    result = []
    for i in range(len(centerPairsIndexes)):
        c1,c2 = centerPairsIndexes[i]
        if c1 is None and c2 is None:
            assert(False)
        elif c2 is None:
            result.append((centers1[c1],None))
        elif c1 is None:
            result.append((None,centers2[c2]))
        else:
            result.append((centers1[c1],centers2[c2]))
    return result

def compare_center_distance(detections1,detections2,radius,distanceMeasure):
    mismatches = 0
    pairs = getCenterPairs(detections1,detections2)
    for i in pairs:
        if (i[0] is None) or (i[1] is None):
            mismatches += 1
            continue
        distance = distanceMeasure(i[0],i[1])
        if (distance > radius):
            mismatches += 1
    return mismatches

def compare_predictions(detections1,detections2,threshold=0.,radius=50,distanceMeasure = Euclidean):
    det1 = [i for i in detections1 if i.conf > threshold]
    det2 = [i for i in detections2 if i.conf > threshold]
    numbersMatch = compare_numbers(det1,det2)
    box_accuracy = compare_center_distance(det1,det2,radius,distanceMeasure)
    return numbersMatch,box_accuracy

import numpy as np
x = np.arange(0, 1, 0.05)
y = np.arange(5, 2000, 15)
xx, yy = np.meshgrid(x, y)
z1 = np.zeros(xx.shape)
z2 = np.zeros(xx.shape)
for i in range(len(y)):
    for j in range(len(x)):
        numbersTotal = 0
        box_accuracyTotal = 0
        for k in range(min(len(imagePreds),len(imagePreds2))):
            numbers, box_accuracy = compare_predictions(imagePreds[k].detections,imagePreds2[k].detections,threshold=xx[i,j],radius=yy[i,j])
            numbersTotal += numbers
            box_accuracyTotal += box_accuracy
        z1[i,j] = numbersTotal
        z2[i,j] = box_accuracyTotal

import matplotlib.pyplot as plt
plt.figure()
h1 = plt.contourf(x,y,z1)
plt.show()
plt.figure()
h2 = plt.contourf(x,y,z2)
plt.show()