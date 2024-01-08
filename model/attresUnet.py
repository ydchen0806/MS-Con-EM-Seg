import torch
import torch.nn as nn
from utils_resnet3d import resnet50

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width, d = x.size()
        query = self.query(x).view(batch_size, -1, height * width * d).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width * d)
        attention = self.softmax(torch.bmm(query, key))
        value = self.value(x).view(batch_size, -1, height * width * d) 
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch_size, -1, height, width, d)
        return out + x

class ResUNetWithAttention(nn.Module):
    def __init__(self, cfg, input_size = (64, 64, 64), input_channel = 1, out_channels=14):
        super(ResUNetWithAttention, self).__init__()

        # Load pre-trained ResNet50
        resnet = resnet50()
        self.input_channel = input_channel
        if input_channel != 1:
            resnet.conv1 = nn.Conv3d(self.input_channel,  64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        resnet.fc = nn.Identity()
        # print(resnet)
        self.cfg = cfg
        if cfg.pretrained:
            state_dict = torch.load(cfg.pretrained_path, map_location='cpu')
            # print(state_dict.keys())
            # resnet.load_state_dict(state_dict,strict=False)
            for name, param in state_dict.items():
                if name in resnet.state_dict() and param.size() == resnet.state_dict()[name].size():
                    resnet.state_dict()[name].copy_(param)
                else:
                    print('Skip loading parameter {}, required shape{}, loaded shape{}.'.format(name, resnet.state_dict()[name].size(), param.size()))
            print('Load pretrained model from {} successfully!'.format(cfg.pretrained_path))

        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4

        # Attention modules
        self.attention4 = SelfAttention(1024)
        self.attention3 = SelfAttention(512)
        self.attention2 = SelfAttention(256)
        self.attention1 = SelfAttention(64)

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
        self.output = nn.Conv3d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)
  
        # Decoder with skip connections
        x = self.upconv4(x5) + self.attention4(x4)
        x = self.decoder4(x)
        x = self.upconv3(x) + self.attention3(x3)
        x = self.decoder3(x)
        x = self.upconv2(x) + self.attention2(x2)
        x = self.decoder2(x)
        x = self.upconv1(x) + self.attention1(x1)
        x = self.adapool(x)
        x = self.decoder1(x)
        # Output
        x = self.output(x)

        return x


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--pretrained_path', type=str, default='/braindat/lab/chenyd/LOGs/Neurips23_imgSSL/barlowTwins0414_final2/checkpoint_35999.pth')
    cfg = parser.parse_args()

    # Test model
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = ResUNetWithAttention(cfg, input_size=(24, 224, 224)).to(device)
    x = torch.randn(1, 1, 24, 224, 224).to(device)
    y = model(x)
    print(y.shape)
    torch.cuda.empty_cache()