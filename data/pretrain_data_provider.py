import monai
import monai.transforms as mt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import os

class random_apply(nn.Module):
    def __init__(self, transform, p):
        super().__init__()
        self.transform = transform
        self.p = p

    def forward(self, x):
        if torch.rand(1) < self.p:
            return self.transform(x)
        return x


def nii_dataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = sorted(glob(os.path.join(data_path, '*.gz')))
        self.transform = transform
        self.data_dict = [{'image': image_name} for image_name in self.data]
        
        self.transform = mt.compose([mt.LoadImaged(keys=["image"]),
                                    mt.EnsureChannelFirstd(keys=["image"]),
                                    mt.ScaleIntensityRanged(
                                        keys=["image"], a_min=100, a_max=255,
                                        b_min=0.0, b_max=1.0, clip=True,
                                    ),
                                    mt.CropForegroundd(keys=["image"], source_key="image"),
                                    mt.Orientationd(keys=["image"], axcodes="RAS"),
                                    # transforms.RandAffined(
                                    #     keys=['image'],
                                    #     mode=('bilinear', 'nearest'),
                                    #     shear_range=(0.5, 0.5, 0.5),
                                    #     prob=1.0, spatial_size=(128,128,48),
                                    #     rotate_range=(0, np.pi/15),
                                    #    ),
                                    mt.RandSpatialCropSamplesd(
                                        keys=["image"],
                                        roi_size=[128,128,48],
                                        random_size=False,
                                        num_samples=8,
                                    ),
                                    ]
                                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        if self.transform:
            data = self.transform(data)
        return data

    def read_nii(self, file_path):
        data = monai.data.load_nifti(file_path)
        return data