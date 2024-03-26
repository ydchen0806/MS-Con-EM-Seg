import torch
import pandas as pd
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
from torchvision.transforms import transforms
from PIL import Image
import nibabel as nib
import os
import copy
from tqdm import tqdm
import monai
from glob import glob
from monai.data import (
    DataLoader,
    CacheDataset,
)
import skimage
from monai.transforms import (
    Compose,
    Orientationd,
    ScaleIntensityRanged,
    RandSpatialCropSamplesd,
    Resized,
    RandRotate90d,
    RandShiftIntensityd,
    ToTensord,
    RandCoarseDropoutd,
    RandCoarseShuffled
)

pd.options.mode.chained_assignment = None

def nifty2numpy(nifti_path):
    img = nib.load(nifti_path)
    return np.array(img.dataobj)


class CT_Report_3D_dataset(Dataset):
    def __init__(self, csv, transform=None, **args):
        self.csv = csv
        self.ct_list = self.csv['path'].to_list()
        self.ct_list = [{'image': q} for q in self.ct_list]
        self.transform = transform
        self.mode = args['train_test']
        self.ct_dir = args['ct_dir']
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        # add your dir here
        # ct_path = os.path.join(self.ct_dir, self.ct_list[idx]['image'])
        ct_path = glob(os.path.join(self.ct_dir,'CT*','ct',self.ct_list[idx]['image']))[0]
        ct_patch = nifty2numpy(ct_path)
        
        ct_patch = torch.tensor(ct_patch)
        ct_patch = ct_patch.unsqueeze(0)
        ct_patch = {'image': ct_patch,
                    'ori_image': ct_patch}
        ct_patch = monai.transforms.ToMetaTensord(keys=['image', 'ori_image'])(ct_patch)
        ct_patch = self.transform(ct_patch)

        aug_ct_patch = [i['image'] for i in ct_patch]
        aug_ct_patch = torch.concat(aug_ct_patch, dim=0)

        ori_ct_patch = [i['ori_image'] for i in ct_patch]
        ori_ct_patch = torch.concat(ori_ct_patch, dim=0)

        data = {
            'ct_patch': aug_ct_patch,
            'ori_ct_patch': ori_ct_patch
        }
        return data

class VLP_dataset:

    def __init__(self, csv_path, ct_dir):
        self.csv_path = csv_path
        self.ct_dir = ct_dir

    def get_dataset(self, train_test, T=None):
        patch_size = [128,128,64]   
        if train_test == 'train':
            print('Apply Train-stage Transform!')

            Transforms = Compose(
                [
                    Orientationd(keys=["image", "ori_image"], axcodes="RAS"),  
                    ScaleIntensityRanged(
                        keys=["image", "ori_image"],
                        a_min=-175,
                        a_max=250,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True,
                    ),
                    RandSpatialCropSamplesd(
                            keys=["image", "ori_image"],
                            roi_size=[patch_size[0]*1, patch_size[1]*1, patch_size[2]*1],
                            num_samples=2,
                            random_center=True,
                            random_size=False,
                        ),
                    RandCoarseDropoutd(
                        keys=["image"],
                        holes = 18,
                        spatial_size = (16, 16, 16),
                        dropout_holes=True,
                        fill_value=None,
                        max_holes=None,
                        max_spatial_size=None,
                        prob=1
                    ),
                    RandCoarseShuffled(
                        keys=["image"],
                        holes = 18,
                        spatial_size = (16, 16, 16),
                        prob = 1
                    ),
                    RandRotate90d(
                            keys=["image"],
                            prob=0.10,
                            max_k=3,
                        ),
                    RandShiftIntensityd(
                        keys=["image"],
                        offsets=0.10,
                        prob=0.5,
                    ),
                    ToTensord(keys=["image", "ori_image"]),
                ]
            )
        else:
            print('Apply Test-stage Transform Beta!')

            Transforms = Compose(
                [
                    Orientationd(keys=["image"], axcodes="RAS"),  
                    ScaleIntensityRanged(
                        keys=["image"],
                        a_min=-175,
                        a_max=250,
                        b_min=0.0,
                        b_max=1.0,
                        clip=True,
                    ),
                    RandSpatialCropSamplesd(
                            keys=["image"],
                            roi_size=[patch_size[0]*1, patch_size[1]*1, patch_size[2]*1],
                            num_samples=3,
                            random_center=True,
                            random_size=False,
                        ),
                ]
            )

        csv = pd.read_csv(self.csv_path)

        misc_args = {'train_test': train_test,
                     'ct_dir': self.ct_dir}

        dataset = CT_Report_3D_dataset(csv=csv,
                                       transform=Transforms,
                                       **misc_args)

        return dataset
