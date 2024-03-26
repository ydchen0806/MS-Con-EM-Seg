import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from monai.networks.nets import SwinUNETR
import sys
sys.path.append('/data/ydchen/VLP/EM_Mamba/SegMamba')
from model_segmamba.segmamba import MambaEncoder, MambaLayer
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock

class SegMamba(nn.Module):
    def __init__(
        self,
        in_chans=1,
        out_chans=13,
        depths=[2, 2, 2, 2],
        feat_size=[48, 96, 192, 384],
        drop_path_rate=0,
        layer_scale_init_value=1e-6,
        hidden_size: int = 768,
        norm_name = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        spatial_dims=3,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.spatial_dims = spatial_dims
        self.mamba_encoder = MambaEncoder(in_chans, 
                              )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_chans,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # self.decoder5 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.hidden_size,
        #     out_channels=self.feat_size[3],
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        # self.decoder4 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.feat_size[3],
        #     out_channels=self.feat_size[2],
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        # self.decoder3 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.feat_size[2],
        #     out_channels=self.feat_size[1],
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        # self.decoder2 = UnetrUpBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.feat_size[1],
        #     out_channels=self.feat_size[0],
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        # self.decoder1 = UnetrBasicBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=self.feat_size[0],
        #     out_channels=self.feat_size[0],
        #     kernel_size=3,
        #     stride=1,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )
        # self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=48, out_channels=self.out_chans)

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        outs = self.mamba_encoder(x_in)
        enc1 = self.encoder1(x_in)
        x2 = outs[0]
        enc2 = self.encoder2(x2)
        x3 = outs[1]
        enc3 = self.encoder3(x3)
        x4 = outs[2]
        enc4 = self.encoder4(x4)
        enc_hidden = self.encoder5(outs[3])
        # dec3 = self.decoder5(enc_hidden, enc4)
        # dec2 = self.decoder4(dec3, enc3)
        # dec1 = self.decoder3(dec2, enc2)
        # dec0 = self.decoder2(dec1, enc1)
        # out = self.decoder1(dec0)
                
        # return self.out(out)
        return enc_hidden

class MambaPool3d(nn.Module):
    def __init__(self, dim = 768):
        super().__init__()
        self.mamba = MambaLayer(dim)
        self.dim = dim
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
    def forward(self, x):
        assert len(x.shape) == 5, f"Input shape must be (B, D, C, H, W), got {x.shape}"
        B, C, D, H, W = x.shape
        assert C == self.dim
        # x = self.mamba(x)
        x = self.pool(x)
        return x

class AttentionPool3d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 3 + 1, embed_dim) / embed_dim ** 0.5)

        self.mhsa = nn.MultiheadAttention(embed_dim, num_heads)        
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.permute(1, 0, 2)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, att_map = self.mhsa(x[:1], x, x, average_attn_weights=True)
        x = self.c_proj(x)
        return x.squeeze(0), att_map.squeeze(0)


class MambaAE(torch.nn.Module):
    def __init__(self, network_config, device_id):
        super(MambaAE, self).__init__()
        self.device_id=device_id
        self.network_config = network_config

        self.proj_hidden = network_config['projection_head']['mlp_hidden_size']
        self.proj_out = network_config['projection_head']['projection_size']

        self.img_model = SegMamba(in_chans=1)
        self.mambapool = MambaPool3d(dim=768)
        self.proj_head = nn.Sequential(
            nn.Linear(self.proj_hidden, self.proj_hidden),
            nn.ReLU(),
            nn.Linear(self.proj_hidden, self.proj_out)
        )
        self.pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, img):
        # split one batch into two views, view1 and view2 are from the same volume
        view1, view2 = img[:, 0, :, :], img[:, 1, :, :]
        view1 = view1.unsqueeze(1)
        view2 = view2.unsqueeze(1)

        view1_emb = self.img_model(view1)
        view2_emb = self.img_model(view1)

        view1_pool_emb = self.mambapool(view1_emb).reshape(view1_emb.shape[0], -1)
        view2_pool_emb = self.mambapool(view2_emb).reshape(view2_emb.shape[0], -1)
        
        proj_view1 = self.proj_head(view1_pool_emb)
        proj_view2 = self.proj_head(view2_pool_emb)


        output_dict = {'view1_proj_img_emb': proj_view1,
                       'view2_proj_img_emb': proj_view2,
                       'view1_img_dec': view1_pool_emb,
                       'view2_img_dec': view2_pool_emb,
                       }
        
        return output_dict