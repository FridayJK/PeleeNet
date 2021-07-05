import argparse 
import os 
import shutil 
import time 
import math 
import sys
import random
import shutil
from glob import glob 

import numpy as np
from scipy import io
#---------------------------------------
"""
move valimg to correspongding folders.
val_id(start from 1) -> ILSVRC_ID(start from 1) -> WIND
"""
#---------------------------------------

# image_path = "./data/ILSVRC2012_val/"
# image_path = "/mnt/data/ILSVRC2012/ILSVRC2012_val/val/*/"
image_path = "/mnt/data/JiaoTongDet/"

# devkit_dir = "/mnt/data/ILSVRC2012/ILSVRC2012_devkit_t12/"
# synset = io.loadmat(os.path.join(devkit_dir, 'data', 'meta.mat'))
# labels = np.loadtxt(os.path.join(devkit_dir, 'data', 'ILSVRC2012_validation_ground_truth.txt'),dtype="int")
# labels = np.loadtxt(devkit_dir + "ILSVRC2012_validation_ground_truth.txt",dtype="int")
images_list = glob(image_path + "*.jpg", recursive=True)

root_data_path = "./data/val_det/"
os.makedirs(root_data_path, exist_ok=True)
images_list = random.sample(images_list, 1000)
for i, img_file in enumerate(images_list):
#     data_path = root_data_path + str(label)
#     os.makedirs(data_path, exist_ok=True)
#     img_path = image_path + "ILSVRC2012_val_" + str(i+1).zfill(8) + ".JPEG"
    # shutil.move(img_file, root_data_path)
    shutil.copy(img_file, root_data_path)
#     if((i+1)%1000==0):
#         print("processd:{}".format(i))

def move_valimg(val_dir='./val', devkit_dir='./ILSVRC2012_devkit_t12'):
    """
    move valimg to correspongding folders.
    val_id(start from 1) -> ILSVRC_ID(start from 1) -> WIND
    organize like:
    /val
       /n01440764
           images
       /n01443537
           images
        .....
    """
    # load synset, val ground truth and val images list
    synset = io.loadmat(os.path.join(devkit_dir, 'data', 'meta.mat'))
    
    ground_truth = open(os.path.join(devkit_dir, 'data', 'ILSVRC2012_validation_ground_truth.txt'))
    lines = ground_truth.readlines()
    labels = [int(line[:-1]) for line in lines]
    
    root, _, filenames = next(os.walk(val_dir))
    for filename in filenames:
        # val image name -> ILSVRC ID -> WIND
        val_id = int(filename.split('.')[0].split('_')[-1])
        ILSVRC_ID = labels[val_id-1]
        WIND = synset['synsets'][ILSVRC_ID-1][0][1][0]
        print("val_id:%d, ILSVRC_ID:%d, WIND:%s" % (val_id, ILSVRC_ID, WIND))
 
        # move val images
        output_dir = os.path.join(root, WIND)
        if os.path.isdir(output_dir):
            pass
        else:
            os.mkdir(output_dir)
        shutil.move(os.path.join(root, filename), os.path.join(output_dir, filename))

# move_valimg("/mnt/data/ILSVRC2012/ILSVRC2012_val/val/","/mnt/data/ILSVRC2012/ILSVRC2012_devkit_t12/")





