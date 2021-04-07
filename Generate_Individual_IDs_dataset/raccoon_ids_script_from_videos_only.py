# -*- coding: utf-8 -*-
"""
file: raccoon_ids_script.py

@author: Suhail.Alnahari

@description: 

@created: 2021-04-06T10:48:08.072Z-05:00

@last-modified: 2021-04-06T16:50:40.059Z-05:00
"""

# standard library
# 3rd party packages
# local source

import pandas as pd
import numpy as np
from typing import List, Dict
import datetime

class Frame:
    def __init__(self, boxes, classifications, name, video):
        self.boxes: List[List[float]] = boxes
        self.classifications: List[str] = classifications
        self.name: str = name
        self.video: str = video

    def _oneAnimalPerFrame(self) -> bool:
        return len(self.classifications) == 1

    def oneAnimalPerFrameGetter(self) -> str:
        if self._oneAnimalPerFrame():
            return self.classifications[-1]
        return ""

    def getDominantAnimal(self) -> str:
        if (len(self.classifications) == 0):
            return ""
        def most_frequent(lst: List[str]) -> str:
            counter = 0
            num = lst[0]
            for i in lst:
                curr_frequency = lst.count(i)
                if(curr_frequency> counter):
                    counter = curr_frequency
                    num = i
  
            return num
        return most_frequent(self.classifications)
    
    def addLabels(self,box: List[float], className: str):
        assert len(box) == 4
        assert len(className) > 0
        self.boxes.append(box)
        self.classifications.append(className)

    def __len__(self) -> int:
        return len(self.classifications)

    def __repr__(self) -> str:
        return f"{self.name} has {len(self.classifications)} classifications with {self.getDominantAnimal()} dominant"

    def __str__(self) -> str:
        return f"{self.name} has {len(self.classifications)} classifications with {self.getDominantAnimal()} dominant"

class Video:
    def __init__(self, frames: List[Frame], name: str,nameFormat: str = '%m%d%Y'):
        for i in frames:
            assert i.video == name
        self.frames: List[Frame] = frames
        self.name: str = name
        self.getDateFromName(nameFormat)

    def isOneSpeciesVideo(self) -> bool:
        animals: Dict[str,int]= {}
        for frame in self.frames:
            if (frame.oneAnimalPerFrameGetter() != ""):
                if (frame.oneAnimalPerFrameGetter() in animals.keys()):
                    animals[frame.oneAnimalPerFrameGetter()] += 1
                else:
                    animals[frame.oneAnimalPerFrameGetter()] = 1
        # print(animals)
        return len(animals.keys()) == 1
    
    def addFrame(self, frame: Frame):
        assert frame.video == self.name
        if not(frame.name in [f.name for f in self.frames]):
            self.frames.append(frame)
    
    def getDominantSpecies(self) -> str:
        if (len(self.frames) == 0):
            return ""
        def most_frequent(lst: List[str]) -> str:
            counter = 0
            num = lst[0]
            for i in lst:
                curr_frequency = lst.count(i)
                if(curr_frequency> counter):
                    counter = curr_frequency
                    num = i
  
            return num
        species: List[str] = []
        for frame in self.frames:
            species.append(frame.getDominantAnimal())
        return most_frequent(species)

    def getAllSpecies(self) -> List:
        if (len(self.frames) == 0):
            return []        
        species: List[str] = []
        for frame in self.frames:
            species.append(frame.getDominantAnimal())
        return np.unique(species)
    
    def __len__(self):
        return len(self.frames)

    def getDateFromName(self,nameFormat):
        fileName = self.name.split('/')[-1]
        date = fileName.split('_')[-3]
        # print(date)
        self.date = datetime.datetime.strptime(date, nameFormat)

    def __repr__(self) -> str:
        result = ''
        for f in self.frames:
            for i in range(len(f)):
                result += ','.join([f.name.split('/')[-1],*[str(b) for b in f.boxes[i]],f.classifications[i]])
                result += '\n'
        return result

    def __str__(self) -> str:
        result = ''
        for f in self.frames:
            for i in range(len(f)):
                result += ','.join([f.name.split('/')[-1],*[str(b) for b in f.boxes[i]],f.classifications[i],self.name.split('/')[-1]])
                result += '\n'
        return result


labelPath = "d:/Zoon_parent/Zooniverse/data/raccoons/university-of-wyoming-raccoon-project-aggregated-for-retinanet.csv"
# labelPath = 'test.csv'

labels = pd.read_csv(labelPath,header=None)

labels['videos'] = labels[0].apply(lambda name: '_'.join(name.split('_')[:-1]))
        
videoList: Dict[str,Video] = {}
frameList: Dict[str, Frame] = {}
for i in labels.values:
    if not(i[-1] in videoList.keys()):
        videoList[i[-1]] = Video([],i[-1])
    if not(i[0] in frameList.keys()):
        frameList[i[0]] = Frame([],[],i[0],i[-1])
        videoList[i[-1]].addFrame(frameList[i[0]])
    frameList[i[0]].addLabels(i[1:5],i[5])
   
print(f"Number of videos in this dataset: {len(videoList)}")
datasetSize = 0
for i in videoList.keys():
    datasetSize += len(videoList[i])
print(f"Number of frames in this dataset: {datasetSize}")

datasetClassificationSize = 0
for i in frameList.keys():
    datasetClassificationSize += len(frameList[i])
print(f"Number of classifications in this dataset: {datasetClassificationSize}")

datasetClassificationSizeVideos = 0
for i in videoList.keys():
    for f in videoList[i].frames:
        datasetClassificationSizeVideos += len(f)
print(f"Number of classifications in this dataset from videos: {datasetClassificationSizeVideos}")


oneSpeciesVideos: Dict[str,Video] = {}
for video in videoList.keys():
    if (videoList[video].isOneSpeciesVideo()):
        oneSpeciesVideos[video] = videoList[video]

species: Dict[str,str] = {}
for video in oneSpeciesVideos:
    species[video] = videoList[video].getDominantSpecies()

onlyTheseSpecies = ['raccoon']

filteredVideos: Dict[str,str] = {}
for i in species.keys():
    if (species[i].lower() in onlyTheseSpecies):
        filteredVideos[i] = species[i]

# print(filteredVideos)     
print(f"Number of videos that have one species of the following categories {onlyTheseSpecies}: {len(filteredVideos)}")
datasetSize = 0
for i in filteredVideos:
    datasetSize += len(videoList[i])

print(f"Number of frames that have one species of the following categories {onlyTheseSpecies}: {datasetSize}")
print(f"average number of frames per video: {sum([len(videoList[i]) for i in filteredVideos])/len(filteredVideos.keys())}")


selectedVideos = [videoList[i] for i in filteredVideos]
selectedVideos = sorted(selectedVideos,key=lambda k: k.date)
selectedVideos = [selected for selected in selectedVideos]

selectedVideosWithThresh = [i.name for i in selectedVideos]
before = len(selectedVideosWithThresh)
after = 0
while(before != after):
    result = []
    before = len(selectedVideosWithThresh)
    deleted = False
    for i in range(len(selectedVideos)-1):
        if (abs((selectedVideos[i].date-selectedVideos[i+1].date).days) < 2):
            if (len(selectedVideos[i]) > len(selectedVideos[i+1])):
                if (not(deleted)):
                    selectedVideosWithThresh.remove(selectedVideos[i+1].name)
            else:
                if (not(deleted)):
                    selectedVideosWithThresh.remove(selectedVideos[i].name)
            deleted = True
        result.append(abs((selectedVideos[i].date-selectedVideos[i+1].date).days))
    selectedVideos = [videoList[selected] for selected in selectedVideosWithThresh]
    after = len(selectedVideosWithThresh)


# import matplotlib.pyplot as plt
# n, bins, patches = plt.hist(x=[len(videoList[i.name]) for i in selectedVideos], bins='auto', color='#0504aa',
#                             alpha=0.7, rwidth=0.85)
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('Number of frames')
# plt.ylabel('Number of videos')
# plt.title('Number of frames for videos where only one raccoon was identified')
# maxfreq = n.max()
# # Set a clean upper y-axis limit.
# plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
# plt.show()


# import matplotlib.pyplot as plt
# n, bins, patches = plt.hist(x=[len(videoList[i]) for i in filteredVideos], bins='auto', color='#0504aa',
#                             alpha=0.7, rwidth=0.85)
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('Number of frames')
# plt.ylabel('Number of videos')
# plt.title('Number of frames for videos where only one raccoon was identified')
# maxfreq = n.max()
# # Set a clean upper y-axis limit.
# plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
# plt.show()


# import matplotlib.pyplot as plt
# n, bins, patches = plt.hist(x=result, bins='auto', color='#0504aa',
#                             alpha=1, rwidth=0.85)
# plt.grid(axis='y', alpha=0.75)
# plt.xlabel('Running difference between dates starting from the earliest date')
# plt.ylabel('Number of videos')
# plt.title('Number of videos where only one raccoon was identified')
# maxfreq = n.max()
# # Set a clean upper y-axis limit.
# plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
# plt.show()


fle = open("videoDataset.csv",'w')
for i in selectedVideos:
    fle.write(str(i))
fle.close()

# for i in videoList.keys():
#     print(f"{i} has these species {videoList[i].getAllSpecies()}")
    