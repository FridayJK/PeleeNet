import argparse 
import os 
import shutil 
import time 
import math 
import sys 

import numpy as np

import torch 
import torch.nn as nn 
import torch.nn.parallel 
import torch.backends.cudnn as cudnn 
import torch.distributed as dist 
import torch.optim 
import torch.utils.data 
import torch.utils.data.distributed 
import torchvision.transforms as transforms 
import torchvision.datasets as datasets 

from peleenet import PeleeNet 
import onnx
print(torch.__version__)

import torchvision
import torch
from torch.autograd import Variable
import onnx
print(torch.__version__)

# input_name = ['input']
# output_name = ['output']
# input = Variable(torch.randn(1, 3, 224, 224)).cuda()
# model = torchvision.models.resnet50(pretrained=True).cuda()
# torch.onnx.export(model, input, 'resnet50.onnx', input_names=input_name, output_names=output_name, verbose=True)

def create_model(weights, num_classes=1000, arch='peleenet',engine='torch'):
    if engine == 'torch':
        if arch == 'peleenet':
                model = PeleeNet(num_classes=num_classes)
        else:
                model = PeleeNet(num_classes=num_classes)
        print(model)
        # model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()
        # model.module.named_parameters
        if os.path.isfile(weights):
            print("=> loading checkpoint '{}'".format(weights))
            checkpoint = torch.load(weights)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(weights))
        # cudnn.benchmark = True
    model = model.module
    return model

weights = "./weights/peleenet_acc7208.pth"
model = create_model(weights)

input_name = ['input']
output_name = ['output']
input = torch.randn(16, 3, 224, 224, device='cuda')
torch.onnx.export(model, input, 'peleeNetBatch16.onnx', input_names=input_name, output_names=output_name, verbose=True)

test = onnx.load('peleeNetBatch16.onnx')
onnx.checker.check_model(test)
print("==> Passed")