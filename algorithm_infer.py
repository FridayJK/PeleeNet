from __future__ import print_function
# import pyshadow
import argparse
import math
import numpy as np
import cv2
import json
import time
import os
import shutil
import pyalgorithm
from eval_trt_mix_precision_det import *

img_root_path = "/mnt/data/JiaoTongDet2/"
image_name = "2019-08-12-07-55-0114.jpg"

module_handle = pyalgorithm.ModuleHandle('/mnt/data/model_fp16_True_int8_False_maxbatch16_peleeDet_atss_testV11_v11-v0.28.tronmodel')
module = module_handle.load_detect(
    enable_profiler=True,
    net_id=0,
    method='fcos',
    backend_type='TensorRT',
    device_id=0,
    max_batch_size=16
)

result = module.run(data=[cv2.imread(img_root_path+image_name),cv2.imread(img_root_path+image_name),cv2.imread(img_root_path+image_name),cv2.imread(img_root_path+image_name),cv2.imread(img_root_path+image_name),
cv2.imread(img_root_path+image_name),cv2.imread(img_root_path+image_name),cv2.imread(img_root_path+image_name),cv2.imread(img_root_path+image_name),cv2.imread(img_root_path+image_name),
cv2.imread(img_root_path+image_name),cv2.imread(img_root_path+image_name),cv2.imread(img_root_path+image_name),cv2.imread(img_root_path+image_name),cv2.imread(img_root_path+image_name),
cv2.imread(img_root_path+image_name)])

# result = module.run(data=[cv2.imread(img_root_path+image_name)])

print(result.shape)
