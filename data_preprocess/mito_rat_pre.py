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

if __name__ == '__main__':
    K_path = '/braindat/large-scale-EM-data/MitoEM/rat/im'
    K_list = sorted(glob(K_path + '/*png'))
    K_list_middle = K_list[len(K_list) // 2 - 500 : len(K_list) // 2 + 500]

    save_path = '/braindat/lab/chenyd/DATASET/miccai_pretrain_data'
    save_name = 'Mito_rat'
    save_path = os.path.join(save_path, save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    k = 0
    temp_hdf = [np.zeros((100, 1024, 1024)) for _ in range(16)]

    for i in tqdm(range(len(K_list_middle))):
        img, num = read_img(K_list_middle[i])
        for j in range(num):
            #print(img[j].shape)
            temp_hdf[j][k] = img[j]
            if k == 99:
                for l in range(16):
                    save_h5(temp_hdf[l], save_path, save_name + '_' + str(i) + '_' + str(l) + '.hdf')
                k = 0
        k += 1
