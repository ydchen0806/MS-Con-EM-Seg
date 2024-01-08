import sys
sys.path.append('/braindat/lab/chenyd/code/Miccai23')
from model.model_superhuman import UNet_PNI_Noskip, UNet_PNI
import yaml
from attrdict import AttrDict

from utils.show import show_one
from dataloader.data_provider_pretraining import Train
import numpy as np
from PIL import Image
import random
import time
import os

import torch
import torch.nn as nn

import numpy as np
# input = np.random.random((1,1,18,160,160)).astype(np.float32)
# x = torch.tensor(input)

x = torch.randn((1,1,18,160,160))

# model = UNet_PNI(filters=[28, 36, 48, 64, 80], upsample_mode='transposeS', merge_mode='cat').to('cuda:0')
# model_structure(model)
# out = model(x)
# print(out.shape)
# from torchstat import stat
# from ptflops import get_model_complexity_info
# model = UNet_PNI(filters=[28, 36, 48, 64, 80], upsample_mode='bilinear', merge_mode='add')
# model = UNet_PNI_embedding(filters=[28, 36, 48, 64, 80], upsample_mode='bilinear', merge_mode='add')
# model = UNet_PNI(filters=[14, 18, 24, 32, 40], upsample_mode='bilinear', merge_mode='add')
# model = UNet_PNI(filters=[18, 26, 38, 54, 70], upsample_mode='bilinear', merge_mode='add')
# model = UNet_PNI(filters=[12, 20, 32, 48, 64], upsample_mode='bilinear', merge_mode='add')
# stat(model, (1, 18, 160, 160))

# macs, params = get_model_complexity_info(model, (1, 18, 160, 160), as_strings=True,
#                                        print_per_layer_stat=True, verbose=True)
# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
# print('{:<30}  {:<8}'.format('Number of parameters: ', params))
model = UNet_PNI_Noskip(show_feature=True)
model2 = UNet_PNI(merge_mode='add', show_feature=True)
k = 0
for name,weight in model.named_parameters():
    if weight.shape == model2.state_dict()[name].shape:
        print(k,'yes')
    else:
        print(k,weight.shape,model2.state_dict()[name].shape)
    k+=1

_,_,upfeature,out = model(x)
# print(out1.shape)
# print(out2.shape)
# print(out3.shape)
# print(out4.shape)
for i in upfeature:
    print(i.shape)

torch.cuda.empty_cache()