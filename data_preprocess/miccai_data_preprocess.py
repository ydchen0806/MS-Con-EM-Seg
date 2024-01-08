import numpy 
import os
from multiprocessing import Pool
from glob import glob
from tqdm import tqdm
import h5py
import SimpleITK as sitk


if __name__ == '__main__':
    path = '/braindat/large-scale-EM-data'
    data_list = glob(path + '/*')
    data_path = []
    for i in data_list:
        data_path.append(sorted(glob(i + '/*')))
    print(data_path[0])