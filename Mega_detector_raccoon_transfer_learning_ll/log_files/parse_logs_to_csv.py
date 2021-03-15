# -*- coding: utf-8 -*-
"""
file: parse_logs_to_csv.py

@author: Suhail.Alnahari

@description: 

@created: 2021-03-01T13:30:00.015Z-06:00

@last-modified: 2021-03-15T01:22:09.274Z-05:00
"""

# standard library
import re 

# 3rd party packages
import pandas as pd
import numpy as np

# local source

columns = ["Step","DetectionBoxes_Precision/mAP", "DetectionBoxes_Precision/mAP (large)", "DetectionBoxes_Precision/mAP (medium)","DetectionBoxes_Precision/mAP (small)"
,"DetectionBoxes_Precision/mAP@.50IOU","DetectionBoxes_Precision/mAP@.75IOU","DetectionBoxes_Recall/AR@1","DetectionBoxes_Recall/AR@10","DetectionBoxes_Recall/AR@100",
"DetectionBoxes_Recall/AR@100 (large)", "DetectionBoxes_Recall/AR@100 (medium)","DetectionBoxes_Recall/AR@100 (small)", "Loss/BoxClassifierLoss/classification_loss",
"Loss/BoxClassifierLoss/localization_loss", "Loss/RPNLoss/localization_loss","Loss/RPNLoss/objectness_loss","Loss/total_loss","loss"]
regex = r"INFO:tensorflow:Saving[\s\n]+dict[\s\n]+for[\s\n]+global[\s\n]+step[\s\n]+(\d*?):[\s\n]+DetectionBoxes_Precision\/mAP[\s\n]+=[\s\n]+(.*?),[\s\n]+DetectionBoxes_Precision\/mAP[\s\n]+\(large\)[\s\n]+=[\s\n]+(.*?), DetectionBoxes_Precision\/mAP[\s\n]+\(medium\)[\s\n]+=[\s\n]+(.*?),[\s\n]+DetectionBoxes_Precision\/mAP \(small\)[\s\n]+=[\s\n]+(.*?),[\s\n]+DetectionBoxes_Precision\/mAP@\.50IOU[\s\n]+=[\s\n]+(.*?),[\s\n]+DetectionBoxes_Precision\/mAP@\.75IOU[\s\n]+=[\s\n]+(.*?),[\s\n]+DetectionBoxes_Recall\/AR@1[\s\n]+=[\s\n]+(.*?),[\s\n]+DetectionBoxes_Recall\/AR@10[\s\n]+=[\s\n]+(.*?),[\s\n]+DetectionBoxes_Recall\/AR@100[\s\n]+=[\s\n]+(.*?),[\s\n]+DetectionBoxes_Recall\/AR@100[\s\n]+\(large\)[\s\n]+=[\s\n]+(.*?), DetectionBoxes_Recall\/AR@100[\s\n]+\(medium\)[\s\n]+=[\s\n]+(.*?),[\s\n]+DetectionBoxes_Recall\/AR@100[\s\n]+\(small\)[\s\n]+=[\s\n]+(.*?), Loss\/BoxClassifierLoss\/classification_loss[\s\n]+=[\s\n]+(.*?), Loss\/BoxClassifierLoss\/localization_loss[\s\n]+=[\s\n]+(.*?), Loss\/RPNLoss\/localization_loss[\s\n]+=[\s\n]+(.*?),[\s\n]+Loss\/RPNLoss\/objectness_loss[\s\n]+=[\s\n]+(.*?),[\s\n]+Loss\/total_loss[\s\n]+=[\s\n]+(.*?), .*?,[\s\n]+loss[\s\n]+=[\s\n]+(.*?)\n"
outputFile = "performance_result.csv"

# columns = ["loss","step"]
# regex = r"INFO:tensorflow:loss = (.*), step = (.*) \("
# outputFile = "loss_result.csv"

filenames = ["slurm-1252331.out","slurm-1285335.out","slurm-1315260.out","slurm-1349088.out","slurm-1394929.out",
"slurm-1437179.out","slurm-1469937.out","slurm-1490872.out","slurm-1519722.out","slurm-1541837.out"]

filetext = ""
for i in filenames:
    textfile = open(i, 'r')
    filetext += textfile.read()
    textfile.close()
matches = re.findall(regex, filetext)

df = pd.DataFrame(data=np.asarray(matches),columns=columns)
df.to_csv(outputFile,index=False)

