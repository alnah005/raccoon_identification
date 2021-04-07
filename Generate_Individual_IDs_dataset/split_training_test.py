# -*- coding: utf-8 -*-
"""
file: split_training_test.py

@author: Suhail.Alnahari

@description: 

@created: 2021-04-06T19:03:55.206Z-05:00

@last-modified: 2021-04-06T19:13:06.727Z-05:00
"""

# standard library
# 3rd party packages
# local source

import os
import shutil
import random


#Prompting user to enter number of files to select randomly along with directory
source=input("Enter the Source Directory : ")
dest=input("Enter the Destination Directory : ")
no_of_files=int(input("Enter The Number of Files To Select : "))

print("%"*25+"{ Details Of Transfer }"+"%"*25)
print("\n\nList of Files Moved to %s :-"%(dest))

#Using for loop to randomly choose multiple files
for i in range(no_of_files):
    #Variable random_file stores the name of the random file chosen
    random_file=random.choice(list(sorted([i for i in os.listdir("croppedImages") if (('.png' in i) or ('.jpg' in i) or ('.jpeg' in i))])))
    print("%d} %s"%(i+1,random_file))
    source_file="%s/%s"%(source,random_file)
    dest_file=dest
    #"shutil.move" function moves file from one directory to another
    shutil.move(source_file,dest_file)

print("\n\n"+"$"*33+"[ Files Moved Successfully ]"+"$"*33)