# deployed model without much flexibility
# useful for stand-alone test, model translation, quantization
import torch.nn as nn
import torch.nn.functional as F
import torch

import sys
sys.path.append('/braindat/lab/chenyd/code/Miccai23')
from model.basic import conv3dBlock, upsampleBlock
from model.residual import resBlock_pni
from model.model_para import model_structure

class UNet_PNI(nn.Module):  # deployed PNI model
    # Superhuman Accuracy on the SNEMI3D Connectomics Challenge. Lee et al.
    # https://arxiv.org/abs/1706.00120
    def __init__(self, in_planes=1, 
                    out_planes=1, 
                    filters=[28, 36, 48, 64, 80],    # [28, 36, 48, 64, 80], [32, 64, 128, 256, 512]
                    upsample_mode='transposeS',  # transposeS, bilinear
                    decode_ratio=1, 
                    merge_mode='cat', 
                    pad_mode='zero', 
                    bn_mode='async',   # async or sync
                    relu_mode='elu', 
                    init_mode='kaiming_normal', 
                    bn_momentum=0.001, 
                    do_embed=True,
                    if_sigmoid=True,
                    show_feature=False):
        # filter_ratio: #filter_decode/#filter_encode
        super(UNet_PNI, self).__init__()
        filters2 = filters[:1] + filters
        self.merge_mode = merge_mode
        self.do_embed = do_embed
        self.depth = len(filters2) - 2
        self.if_sigmoid = if_sigmoid
        self.show_feature = show_feature

        # 2D conv for anisotropic
        self.embed_in = conv3dBlock([in_planes], 
                                    [filters2[0]], 
                                    [(1, 5, 5)], 
                                    [1], 
                                    [(0, 2, 2)], 
                                    [True], 
                                    [pad_mode], 
                                    [''], 
                                    [relu_mode], 
                                    init_mode, 
                                    bn_momentum)

        # downsample stream
        self.conv0 = resBlock_pni(filters2[0], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool0 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv1 = resBlock_pni(filters2[1], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = resBlock_pni(filters2[2], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = resBlock_pni(filters2[3], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.center = resBlock_pni(filters2[4], filters2[5], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        # upsample stream
        self.up0 = upsampleBlock(filters2[5], filters2[4], (1,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat0 = conv3dBlock([0], [filters2[4]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat0 = conv3dBlock([0], [filters2[4]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv4 = resBlock_pni(filters2[4]*2, filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up1 = upsampleBlock(filters2[4], filters2[3], (1,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat1 = conv3dBlock([0], [filters2[3]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat1 = conv3dBlock([0], [filters2[3]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv5 = resBlock_pni(filters2[3]*2, filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up2 = upsampleBlock(filters2[3], filters2[2], (1,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat2 = conv3dBlock([0], [filters2[2]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat2 = conv3dBlock([0], [filters2[2]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv6 = resBlock_pni(filters2[2]*2, filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up3 = upsampleBlock(filters2[2], filters2[1], (1,2,2), upsample_mode, init_mode=init_mode)
        if self.merge_mode == 'add':
            self.cat3 = conv3dBlock([0], [filters2[1]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        else:
            self.cat3 = conv3dBlock([0], [filters2[1]*2], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
            self.conv7 = resBlock_pni(filters2[1]*2, filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.embed_out = conv3dBlock([int(filters2[0])], 
                                        [int(filters2[0])], 
                                        [(1, 5, 5)], 
                                        [1], 
                                        [(0, 2, 2)], 
                                        [True], 
                                        [pad_mode], 
                                        [''], 
                                        [relu_mode], 
                                        init_mode, 
                                        bn_momentum)

        self.out_put = conv3dBlock([int(filters2[0])], [out_planes], [(1, 1, 1)], init_mode=init_mode)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # embedding
        embed_in = self.embed_in(x)
        conv0 = self.conv0(embed_in)
        pool0 = self.pool0(conv0)
        conv1 = self.conv1(pool0)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        center = self.center(pool3)

        up0 = self.up0(center)
        if self.merge_mode == 'add':
            cat0 = self.cat0(up0 + conv3)
        else:
            cat0 = self.cat0(torch.cat([up0, conv3], dim=1))
        conv4 = self.conv4(cat0)

        up1 = self.up1(conv4)
        if self.merge_mode == 'add':
            cat1 = self.cat1(up1 + conv2)
        else:
            cat1 = self.cat1(torch.cat([up1, conv2], dim=1))
        conv5 = self.conv5(cat1)

        up2 = self.up2(conv5)
        if self.merge_mode == 'add':
            cat2 = self.cat2(up2 + conv1)
        else:
            cat2 = self.cat2(torch.cat([up2, conv1], dim=1))
        conv6 = self.conv6(cat2)

        up3 = self.up3(conv6)
        if self.merge_mode == 'add':
            cat3 = self.cat3(up3 + conv0)
        else:
            cat3 = self.cat3(torch.cat([up3, conv0], dim=1))
        conv7 = self.conv7(cat3)

        embed_out = self.embed_out(conv7)
        out = self.out_put(embed_out)

        if self.if_sigmoid:
            out = torch.sigmoid(out)

        if self.show_feature:
            down_features = [conv0, conv1, conv2, conv3]
            center_features = [center]
            up_features = [conv4, conv5, conv6, conv7]
            return down_features, center_features, up_features, out
        else:
            return out


class UNet_PNI_Noskip(nn.Module):  # deployed PNI model
    # Superhuman Accuracy on the SNEMI3D Connectomics Challenge. Lee et al.
    # https://arxiv.org/abs/1706.00120
    def __init__(self, in_planes=1, 
                    out_planes=1, 
                    filters=[28, 36, 48, 64, 80],    # [28, 36, 48, 64, 80], [32, 64, 128, 256, 512]
                    upsample_mode='bilinear', 
                    decode_ratio=1, 
                    merge_mode='cat', 
                    pad_mode='zero', 
                    bn_mode='async',   # async or sync
                    relu_mode='elu', 
                    init_mode='kaiming_normal', 
                    bn_momentum=0.001, 
                    do_embed=True,
                    if_sigmoid=True,
                    show_feature=False):
        # filter_ratio: #filter_decode/#filter_encode
        super(UNet_PNI_Noskip, self).__init__()
        filters2 = filters[:1] + filters
        self.merge_mode = merge_mode
        self.do_embed = do_embed
        self.depth = len(filters2) - 2
        self.if_sigmoid = if_sigmoid
        self.show_feature = show_feature
        # self.mha = nn.MultiheadAttention(16 ** 3, 4 ,batch_first=True)
        # self.adapool = nn.AdaptiveAvgPool3d((16,16,16))

        
        # 2D conv for anisotropic
        self.embed_in = conv3dBlock([in_planes], 
                                    [filters2[0]], 
                                    [(1, 5, 5)], 
                                    [1], 
                                    [(0, 2, 2)], 
                                    [True], 
                                    [pad_mode], 
                                    [''], 
                                    [relu_mode], 
                                    init_mode, 
                                    bn_momentum)

        # downsample stream
        self.conv0 = resBlock_pni(filters2[0], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool0 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv1 = resBlock_pni(filters2[1], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = resBlock_pni(filters2[2], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = resBlock_pni(filters2[3], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.center = resBlock_pni(filters2[4], filters2[5], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        # upsample stream
        self.up0 = upsampleBlock(filters2[5], filters2[4], (1,2,2), upsample_mode, init_mode=init_mode)
        self.cat0 = conv3dBlock([0], [filters2[4]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
        self.conv4 = resBlock_pni(filters2[4], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up1 = upsampleBlock(filters2[4], filters2[3], (1,2,2), upsample_mode, init_mode=init_mode)
        self.cat1 = conv3dBlock([0], [filters2[3]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
        self.conv5 = resBlock_pni(filters2[3], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up2 = upsampleBlock(filters2[3], filters2[2], (1,2,2), upsample_mode, init_mode=init_mode)
        self.cat2 = conv3dBlock([0], [filters2[2]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
        self.conv6 = resBlock_pni(filters2[2], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.up3 = upsampleBlock(filters2[2], filters2[1], (1,2,2), upsample_mode, init_mode=init_mode)
        self.cat3 = conv3dBlock([0], [filters2[1]], bn_mode=[bn_mode], relu_mode=[relu_mode], bn_momentum=bn_momentum)
        self.conv7 = resBlock_pni(filters2[1], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        self.embed_out = conv3dBlock([int(filters2[0])], 
                                        [int(filters2[0])], 
                                        [(1, 5, 5)], 
                                        [1], 
                                        [(0, 2, 2)], 
                                        [True], 
                                        [pad_mode], 
                                        [''], 
                                        [relu_mode], 
                                        init_mode, 
                                        bn_momentum)

        self.out_put = conv3dBlock([int(filters2[0])], [out_planes], [(1, 1, 1)], init_mode=init_mode)
        self.sigmoid = nn.Sigmoid()

    def encoder(self, x):
        # embedding
        embed_in = self.embed_in(x)
        conv0 = self.conv0(embed_in)
        pool0 = self.pool0(conv0)
        conv1 = self.conv1(pool0)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        center = self.center(pool3)
        down_features = [conv0, conv1, conv2, conv3]

        return center, down_features

    def decoder(self, center):
        up0 = self.up0(center)
        cat0 = self.cat0(up0)
        conv4 = self.conv4(cat0)

        up1 = self.up1(conv4)
        cat1 = self.cat1(up1)
        conv5 = self.conv5(cat1)

        up2 = self.up2(conv5)
        cat2 = self.cat2(up2)
        conv6 = self.conv6(cat2)

        up3 = self.up3(conv6)
        cat3 = self.cat3(up3)
        conv7 = self.conv7(cat3)

        embed_out = self.embed_out(conv7)
        out = self.out_put(embed_out)
        up_features = [conv4, conv5, conv6, conv7]

        return out, up_features

    def forward(self, x):
        center, down_features = self.encoder(x)
        out, up_features = self.decoder(center)

        if self.if_sigmoid:
            out = torch.sigmoid(out)

        if self.show_feature:
            center_features = [center]
            return down_features, center_features, up_features, out
        else:
            return out


class UNet_PNI_encoder(nn.Module):
    def __init__(self, in_planes=1, 
                    filters=[32, 64, 128, 256, 512], 
                    pad_mode='zero', 
                    bn_mode='async',   # async or sync
                    relu_mode='elu', 
                    init_mode='kaiming_normal', 
                    bn_momentum=0.001, 
                    do_embed=True,
                    num_classes=None):
        super(UNet_PNI_encoder, self).__init__()
        filters2 = filters[:1] + filters
        self.do_embed = do_embed
        self.depth = len(filters2) - 2
        self.num_classes = num_classes

        # 2D conv for anisotropic
        self.embed_in = conv3dBlock([in_planes], 
                                    [filters2[0]], 
                                    [(1, 5, 5)], 
                                    [1], 
                                    [(0, 2, 2)], 
                                    [True], 
                                    [pad_mode], 
                                    [''], 
                                    [relu_mode], 
                                    init_mode, 
                                    bn_momentum)

        # downsample stream
        self.conv0 = resBlock_pni(filters2[0], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool0 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv1 = resBlock_pni(filters2[1], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv2 = resBlock_pni(filters2[2], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.conv3 = resBlock_pni(filters2[3], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))

        self.center = resBlock_pni(filters2[4], filters2[5], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        if self.num_classes is not None:
            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc = nn.Linear(filters[-1], num_classes)

    def encoder(self, x):
        # embedding
        embed_in = self.embed_in(x)
        conv0 = self.conv0(embed_in)
        pool0 = self.pool0(conv0)
        conv1 = self.conv1(pool0)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        center = self.center(pool3)
        return center

    def forward(self, x):
        x = self.encoder(x)
        if self.num_classes is not None:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x


if __name__ == "__main__":
    import numpy as np
    import yaml
    from attrdict import AttrDict
    import os
    """"""
    cfg_file = 'pretraining_all.yaml'
    with open(os.path.join('/braindat/lab/chenyd/code/Miccai23/config',cfg_file), 'r') as f:
        cfg = AttrDict( yaml.safe_load(f) )

    input = np.random.random((1,1,36,320,320)).astype(np.float32)
    x = torch.tensor(input).cuda()
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
    model = UNet_PNI_Noskip(in_planes=cfg.MODEL.input_nc,
                                out_planes=cfg.MODEL.output_nc,
                                filters=cfg.MODEL.filters,
                                upsample_mode=cfg.MODEL.upsample_mode,
                                decode_ratio=cfg.MODEL.decode_ratio,
                                merge_mode=cfg.MODEL.merge_mode,
                                pad_mode=cfg.MODEL.pad_mode,
                                bn_mode=cfg.MODEL.bn_mode,
                                relu_mode=cfg.MODEL.relu_mode,
                                init_mode=cfg.MODEL.init_mode,
                                if_sigmoid=cfg.MODEL.if_sigmoid,
                                show_feature=True).cuda()

    out = model(x)
    # print(out1.shape)
    # print(out2.shape)
    # print(out3.shape)
    # print(out4.shape)
    # 查看GPU利用率
    import GPUtil
    GPUtil.showUtilization()
    for i in out:
        try:
            print(i.shape)
        except:
            for k,j in enumerate(i):
                print(k,j.shape)
    
    
    torch.cuda.empty_cache()
    GPUtil.showUtilization()