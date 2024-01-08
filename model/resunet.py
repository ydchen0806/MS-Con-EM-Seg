import torch
import torch.nn as nn
import sys
sys.path.append('/braindat/lab/chenyd/code/Miccai23/model')
from utils_resnet3d import resnet50


class ResUNet(nn.Module):
    def __init__(self, cfg, input_channel = 1, input_size = (64, 64, 64), out_channels=14):
        super(ResUNet, self).__init__()
        # Load pre-trained ResNet50
        resnet = resnet50()
        self.input_channel = input_channel
        self.output_channel = out_channels
        if self.input_channel != 1:
            resnet.conv1 = nn.Conv3d(self.input_channel,  64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        resnet.fc = nn.Identity()
        # print(resnet)
        self.cfg = cfg
        if cfg.pretrained:
            state_dict = torch.load(cfg.pretrained_path, map_location='cpu')
            if not 'barlowT' in cfg.pretrained_path:
                for name, param in state_dict.items():
                    if name in resnet.state_dict() and param.size() == resnet.state_dict()[name].size():
                        resnet.state_dict()[name].copy_(param)
                    else:
                        print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(name, resnet.state_dict()[name].size(), param.size()))
            else:
                state_dict = state_dict['model']
                new_state_dict = {}
                for name, param in state_dict.items():
                    if 'module.backbone.' in name:
                        new_name = name.replace('module.backbone.', '')
                        new_state_dict[new_name] = param
                # print(new_state_dict.keys())
                # print(resnet.state_dict().keys())
                for name, param in new_state_dict.items():
                    if name in resnet.state_dict() and param.size() == resnet.state_dict()[name].size():
                        resnet.state_dict()[name].copy_(param)
                    else:
                        print('Skip loading parameter {},  loaded shape{}.'.format(name, param.size()))
            print('Load pretrained model from {} successfully!'.format(cfg.pretrained_path))
        # Encoder (ResNet50 layers)
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

        # Decoder
        self.upconv4 = nn.ConvTranspose3d(2048, 1024, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(nn.Conv3d(1024, 1024, kernel_size=3, padding=1), nn.ReLU(), nn.Conv3d(1024, 1024, kernel_size=3, padding=1), nn.ReLU())
        self.upconv3 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(nn.Conv3d(512, 512, kernel_size=3, padding=1), nn.ReLU(), nn.Conv3d(512, 512, kernel_size=3, padding=1), nn.ReLU())
        self.upconv2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(nn.Conv3d(256, 256, kernel_size=3, padding=1), nn.ReLU(), nn.Conv3d(256, 256, kernel_size=3, padding=1), nn.ReLU())
        self.upconv1 = nn.ConvTranspose3d(256, 64, kernel_size=(3,7,7), stride=(1, 1, 1), padding=(1, 3, 3))
        self.adapool = nn.AdaptiveAvgPool3d(input_size)
        self.decoder1 = nn.Sequential(nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.ReLU(), nn.Conv3d(64, 64, kernel_size=3, padding=1), nn.ReLU())
   
        # Final output layer
        self.output = nn.Conv3d(64, self.output_channel, kernel_size=1)
    
    def center_crop_padding(self, x, y):
        _, _, d, h, w = x.shape
        _, _, td, th, tw = y.shape
        if d == td and h == th and w == tw:
            return x + y
        elif d < td or h < th or w < tw:
            d1 = (td - d) // 2
            d2 = td - d - d1
            h1 = (th - h) // 2
            h2 = th - h - h1
            w1 = (tw - w) // 2
            w2 = tw - w - w1
            x = nn.functional.pad(x, [w1, w2, h1, h2, d1, d2])
            return x + y
        elif d > td or h > th or w > tw:
            d1 = (d - td) // 2
            d2 = d - td - d1
            h1 = (h - th) // 2
            h2 = h - th - h1
            w1 = (w - tw) // 2
            w2 = w - tw - w1
            x = x[:, :, d1:d-d2, h1:h-h2, w1:w-w2]
            return x + y

    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)
  
        # Decoder with skip connections
        # print(f'upconv4: {self.upconv4(x5).shape}, x4: {x4.shape}')
        x = self.center_crop_padding(self.upconv4(x5), x4) 
        x = self.decoder4(x)
        x = self.center_crop_padding(self.upconv3(x), x3) 
        x = self.decoder3(x)
        x = self.center_crop_padding(self.upconv2(x), x2) 
        x = self.decoder2(x)
        x = self.center_crop_padding(self.upconv1(x), x1) 
        x = self.adapool(x)
        x = self.decoder1(x)
        # Output
        x = self.output(x)

        return torch.sigmoid(x)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--pretrained_path', type=str, default='/braindat/lab/chenyd/MODEL/Neurips_res0429/resnet50_barlowClip/encoder/testclip_bt_resnet_14001_iterations_encoder.pth')
    cfg = parser.parse_args()

    # Test model
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = ResUNet(cfg, input_size=(192, 160, 80),input_channel=1,out_channels=1).to(device)
    x = torch.randn(1, 1, 192, 160, 80).to(device)
    y = model(x)
    print(y.shape)
    torch.cuda.empty_cache()