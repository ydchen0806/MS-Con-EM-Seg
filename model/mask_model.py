import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

### mask scale/mask channel
class Diago_ResNet_MaskScale(torch.nn.Module):
    def __init__(self, mhsa_dim=512):
        super(Diago_ResNet_MaskScale, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=False)

        self.encoder = resnet
        self.encoder.fc = nn.Identity()
        
        self.mhsa_dim = mhsa_dim
        # image emebedding ==> FINDINGS
        self.cls_token = nn.Parameter(torch.zeros((1, 1, 256), dtype=torch.float32))
        self.positional_embedding = nn.Parameter((torch.zeros((1, self.mhsa_dim+1, 256), dtype=torch.float32)))

        self.pool = nn.AdaptiveAvgPool2d((16,16)) # unify the size
        self.flatten = nn.Flatten(2,3)

        self.multi_attent = nn.MultiheadAttention(256, 4, batch_first=True)
        self.read_proj = nn.Sequential(
            nn.Linear(256, 128, bias=True)
        )

        # image embedding + FINDINGS(predicted) ==> IMPRESSION 
        self.diag_proj = nn.Sequential(
            nn.Linear(2048, 2048, bias=True),
            nn.LayerNorm(2048),
            nn.Linear(2048, 128, bias=True)
        )

    def compute(self, x, mask_method='scale'):
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        
        # get multi-scale feature
        fea_1 = self.encoder.layer1(x)
        del x
        fea_2 = self.encoder.layer2(fea_1)
        fea_3 = self.encoder.layer3(fea_2)
        fea_4 = self.encoder.layer4(fea_3)

        img_emb =  self.encoder.avgpool(fea_4)
        img_emb = img_emb.view(img_emb.shape[0], img_emb.shape[1])

        # read image, random mask different ratio channels, flatten to 256 length sequence
        if mask_method == 'scale':
            fea = [fea_1,fea_2,fea_3,fea_4]
            mask_scale_idx = torch.randint(0, 4, (1,))
            find_emb = self.flatten(self.pool(fea[mask_scale_idx]))
            # if mask scale, self.mhsa_dim > 2048
            find_emb = F.pad(find_emb, (0, 0, 0, self.mhsa_dim-find_emb.shape[1], 0, 0), 'constant', 0)

        elif mask_method == 'channel':
            unmask_1 = torch.randint(0, 256, (int(256*0.25),))
            unmask_2 = torch.randint(0, 512, (int(512*0.15),))
            unmask_3 = torch.randint(0, 1024, (int(1024*0.1),))
            unmask_4 = torch.randint(0, 2048, (int(2048*0.1),))

            fea_1 = self.flatten(self.pool(fea_1[:, unmask_1, :, :]))
            fea_2 = self.flatten(self.pool(fea_2[:, unmask_2, :, :]))
            fea_3 = self.flatten(self.pool(fea_3[:, unmask_3, :, :]))
            fea_4 = self.flatten(self.pool(fea_4[:, unmask_4, :, :]))

            # concat all sequence features
            find_emb = torch.cat([fea_1, fea_2, fea_3, fea_4], dim=1)
            find_emb = F.pad(find_emb, (0, 0, 0, self.mhsa_dim-find_emb.shape[1], 0, 0), 'constant', 0)

        # add positional embedding and cls token
        find_emb = find_emb + self.positional_embedding[:, 1:, :]

        self.cls_tokens = self.cls_token + self.positional_embedding[:, :1, :]
        self.cls_tokens = self.cls_tokens.expand(find_emb.shape[0], -1, -1) 
        find_emb = torch.cat([find_emb, self.cls_tokens], dim=1)
        
        # attention operation
        # only extract the cls token emb
        find_emb = self.multi_attent(find_emb, find_emb, find_emb, need_weights=False)[0]+find_emb # output shape is (B, 32, 256)
        find_emb = self.read_proj(find_emb[:, -1, :]) # output shape is (B, 32, 256)
        
        imp_emb = self.diag_proj(img_emb)

        return img_emb, find_emb, imp_emb

    def forward(self, x1, x2, mask_method=['scale', 'channel']):
        if len(set(mask_method)) == 2:
            # check mask method has two different way
            img_emb1_m1, find_emb1_m1, imp_emb1_m1 = self.compute(x1, mask_method=mask_method[0])
            img_emb2_m1, find_emb2_m1, imp_emb2_m1 = self.compute(x2, mask_method=mask_method[0])

            img_emb1_m2, find_emb1_m2, imp_emb1_m2 = self.compute(x1, mask_method=mask_method[1])
            img_emb2_m2, find_emb2_m2, imp_emb2_m2 = self.compute(x2, mask_method=mask_method[1])
        else:
            # if only one mask method, mask_method[idx] can be same
            img_emb1_m1, find_emb1_m1, imp_emb1_m1 = self.compute(x1, mask_method=mask_method[0])
            img_emb2_m1, find_emb2_m1, imp_emb2_m1 = self.compute(x2, mask_method=mask_method[0])

            img_emb1_m2, find_emb1_m2, imp_emb1_m2 = self.compute(x1, mask_method=mask_method[0])
            img_emb2_m2, find_emb2_m2, imp_emb2_m2 = self.compute(x2, mask_method=mask_method[0])
         
        # return img_emeb (b, 2048)
        # find_emb: multi-scale emb (b, 128)
        # imp_emb: last layer emb (b, 128)
        return {'img_emb1_m1': img_emb1_m1, 'find_emb1_m1': find_emb1_m1, 'imp_emb1_m1': imp_emb1_m1,
                'img_emb2_m1': img_emb2_m1, 'find_emb2_m1': find_emb2_m1, 'imp_emb2_m1': imp_emb2_m1,
                'img_emb1_m2': img_emb1_m2, 'find_emb1_m2': find_emb1_m2, 'imp_emb1_m2': imp_emb1_m2,
                'img_emb2_m2': img_emb2_m2, 'find_emb2_m2': find_emb2_m2, 'imp_emb2_m2': imp_emb2_m2}
