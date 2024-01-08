import h5py
import numpy as np
import os
import glob
import SimpleITK as sitk    
# 多线程
from multiprocessing import Pool
from functools import partial
import time

# 计算cpu个数
cpu_num = os.cpu_count()
def process(data):
    hdf_data = h5py.File(data, 'r+')
    temp_data = hdf_data['main'][:]
    if temp_data.dtype != np.uint8:
        temp_data = temp_data.astype(np.uint8)
        del hdf_data['main']
        hdf_data.create_dataset('main', data=temp_data)
        print(temp_data.shape)

def process2(data):
    hdf_data = h5py.File(data, 'r')
    temp_data = hdf_data['main'][:]
    if temp_data.dtype == np.uint8:
        print(data,'ok',temp_data.max())
    else:
        print(data,temp_data.dtype,temp_data.max())
def main():
    # data_path = '/home/zhengyuan/zhengyuan/zhengyuan/data_preprocess_miccai23/data/BRATS2015_Training/HGG'
    data_path = '/braindat/lab/chenyd/DATASET/unlabel_data'
    data_list = glob.glob(data_path + '/*.hdf')
    cpu_num = os.cpu_count() 
    print(len(data_list),cpu_num)
    # data_list = data_list[:10]
    # for data in data_list:
    #     process(data)
    #     print(data)
    #     break
    # print('done')
    pool = Pool(processes=cpu_num // 2)
    for i in data_list:
        # if 'Mi' in i:
        pool.apply_async(process, args=(i, ))
    pool.close()
    pool.join()
    print('done')

if __name__ == '__main__':
    main()
    