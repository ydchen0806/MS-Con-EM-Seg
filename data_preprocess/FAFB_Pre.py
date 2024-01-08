import numpy as np 
import os
from multiprocessing import Pool
from glob import glob
from tqdm import tqdm
import h5py
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm

def save_h5(data, save_path, save_name):
    with h5py.File(os.path.join(save_path, save_name), 'w') as f:
        f.create_dataset('main', data=data,dtype=np.uint8)

def read_img(path, size = 1024):
    img = Image.open(path)
    data = np.array(img)
    x,y = data.shape
    # 从中点取子图
    x = x // 2
    y = y // 2
    img = []
    for i in range(-2, 2):
        for j in range(-2 ,2):
            img.append(data[x + i * size : x + i * size + size, y + j * size : y + j * size + size])
    return img, len(img)