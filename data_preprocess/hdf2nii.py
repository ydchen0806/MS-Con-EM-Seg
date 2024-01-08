import SimpleITK as sitk
import numpy as np
import os
import h5py
import torch
from glob import glob
import torch.nn as nn
import multiprocessing as mp
from multiprocessing import Pool


# read_h5py
def read_h5py(file_path):
    f = h5py.File(file_path, 'r')
    data = f['main'][:]
    f.close()
    return data

# save niigz
def save_niigz(data, file_path):
    data = read_h5py(data)
    print(data.shape)
    data = sitk.GetImageFromArray(data)
    sitk.WriteImage(data, file_path)


if __name__ == '__main__':
    hdf_path = '/braindat/lab/chenyd/DATASET/unlabel_data'
    hdf_data = sorted(glob(os.path.join(hdf_path, '*.hdf')))
    save_path = '/braindat/lab/chenyd/DATASET/unlabel_data_niigz'
    pool = Pool(processes=mp.cpu_count())
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i in hdf_data:
        pool.apply_async(save_niigz, args=(i, os.path.join(save_path, i.split('/')[-1].split('.')[0] + '.nii.gz')))
    pool.close()
    pool.join()
    print('Done!')