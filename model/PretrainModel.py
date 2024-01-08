from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
import yaml
import time
import logging
import argparse
import numpy as np
from attrdict import AttrDict
from tensorboardX import SummaryWriter
import random
import torch
import torch.nn as nn
import sys
sys.path.append('Miccai23')
from model.unet3d_mala import UNet3D_MALA, UNet3D_MALA_ns
from model.model_superhuman import UNet_PNI, UNet_PNI_Noskip
from model.unetr import UNETR
from utils.utils import setup_seed

class pretrainModel(nn.Module):
    def __init__(self, cfg):
        super(pretrainModel, self).__init__()
        self.cfg = cfg
        print('Building model on ', end='', flush=True)
        # t1 = time.time()
        # device = torch.device('cuda:0')
        if self.cfg.MODEL.model_type == 'mala':
            print('load mala model without skip!')
            self.model = UNet3D_MALA_ns(output_nc=self.cfg.MODEL.output_nc, if_sigmoid=self.cfg.MODEL.if_sigmoid, init_mode=self.cfg.MODEL.init_mode_mala, show_feature=True)
        elif self.cfg.MODEL.model_type == 'superhuman':
            print('load superhuman model without skip!')
            self.model = UNet_PNI_Noskip(in_planes=self.cfg.MODEL.input_nc,
                                    out_planes=self.cfg.MODEL.output_nc,
                                    filters=self.cfg.MODEL.filters,
                                    upsample_mode=self.cfg.MODEL.upsample_mode,
                                    decode_ratio=self.cfg.MODEL.decode_ratio,
                                    merge_mode=self.cfg.MODEL.merge_mode,
                                    pad_mode=self.cfg.MODEL.pad_mode,
                                    bn_mode=self.cfg.MODEL.bn_mode,
                                    relu_mode=self.cfg.MODEL.relu_mode,
                                    init_mode=self.cfg.MODEL.init_mode,
                                    if_sigmoid=self.cfg.MODEL.if_sigmoid,
                                    show_feature=True)
        elif self.cfg.MODEL.model_type == 'UNETR':
            print('load UNETR model without skip!')
            self.model = UNETR(
                            in_channels=self.cfg.MODEL.input_nc,
                            out_channels=self.cfg.MODEL.output_nc,
                            img_size=self.cfg.MODEL.unetr_size,
                            patch_size=self.cfg.MODEL.patch_size,
                            feature_size=16,
                            hidden_size=768,
                            mlp_dim=2048,
                            num_heads=8,
                            pos_embed='perceptron',
                            norm_name='instance',
                            conv_block=True,
                            res_block=True,
                            kernel_size=self.cfg.MODEL.kernel_size,
                            skip_connection=False,
                            show_feature=True,
                            dropout_rate=0.1)

        self.mha = nn.MultiheadAttention(16 ** 3, 4 ,batch_first=True)
        self.adapool = nn.AdaptiveAvgPool3d((16,16,16))
		
    def forward(self, inputs1, inputs2, gt):
        if self.cfg.MODEL.model_type == 'mala':
            _,_,up_feature1,out1 = self.model(inputs1)
            _,_,up_feature2,out2 = self.model(inputs2)
            gt = gt[...,14:-14,106:-106,106:-106]
            inputs1 = inputs1[...,14:-14,106:-106,106:-106]
            inputs2 = inputs2[...,14:-14,106:-106,106:-106]
            
        else:
            _,_,up_feature1,out1 = self.model(inputs1)
            _,_,up_feature2,out2 = self.model(inputs2)
        # print('out1',out1.device,'out2',out2.device,'gt',gt.device)
        ##############################
        # LOSS
        # tmp_data = tmp_data[14:-14,106:-106,106:-106]


        for k,(x,y) in enumerate(zip(up_feature1,up_feature2)):
            pool = self.adapool
            x = pool(x)
            y = pool(y)
            if self.cfg.TRAIN.random_choice:
                _, c_x, _, _, _ = x.shape
                _, c_y, _, _, _ = y.shape
                assert c_x == c_y
                
                index_feature = random.sample(range(c_x), int(c_x * 0.1 * (k+1)))
            
                x = x[:,index_feature,:,:,:]
                y = y[:,index_feature,:,:,:]

            x = x.flatten(start_dim=2)
            y = y.flatten(start_dim=2)
            if k == 0:
                x_concat = x
                y_concat = y
            else:
                x_concat = torch.cat([x_concat,x],dim=1)
                y_concat = torch.cat([y_concat,y],dim=1)
       
        x_mha = self.mha(x_concat,y_concat,y_concat)[0]
        y_mha = self.mha(y_concat,x_concat,x_concat)[0]

        return gt, out1, out2, x_concat, y_concat, x_mha, y_mha

if __name__ == '__main__':
    import yaml
    from attrdict import AttrDict
    
    """"""
    seed = 555
    np.random.seed(seed)
    random.seed(seed)
    cfg_file = 'pretraining_all.yaml'
    with open(os.path.join('Miccai23/config',cfg_file), 'r') as f:
        cfg = AttrDict( yaml.safe_load(f) )

    model = pretrainModel(cfg).cuda()
    model = nn.DataParallel(model)
    input1 = torch.randn((1,1,32,160,160)).cuda()
    input2 = torch.randn((1,1,32,160,160)).cuda()
    gt = torch.randn((1,1,32,160,160)).cuda()
    gt, out1, out2, x_concat, y_concat, x_mha, y_mha = model(input1,input2,gt)
    print('out1 shape:',out1.shape,'out2 shape:',out2.shape,'x_concat shape:',\
        x_concat.shape,'y_concat shape:',y_concat.shape,'x_mha shape:',x_mha.shape,'y_mha shape:',y_mha.shape,'gt shape:',gt.shape)
    torch.cuda.empty_cache()
        