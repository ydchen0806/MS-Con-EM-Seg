import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk
import h5py
import time
import torch
# 多线程
from multiprocessing import Pool
from functools import partial



def get_data_list(data_path):
    data_list = glob(data_path + '/*.png')
    return data_list,len(data_list)

def remove_black_border_and_correct(img):
    # 边缘检测
    # edges = cv2.Canny(img, 50, 150)
    _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    # 获取边缘的四个顶点坐标
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    # 对齐坐标系
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # 计算投影变换矩阵
    w = rect[1][0]
    h = rect[1][1]
    src_pts = box.astype("float32")
    dst_pts = np.array([[0, h-1], [0, 0], [w-1, 0], [w-1, h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # 应用变换
    warped = cv2.warpPerspective(img, M, (int(w), int(h)))

    return warped, M

def remove_black_border_and_correct2(img, M):
    # 应用变换
    h, w = img.shape[:2]
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped

def remove_black_border(img):
    _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = img[y:y+h, x:x+w]
    return crop,x,y,w,h

def process(data,save_dir,x,y,w,h):
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    img = cv2.imread(data,0)
    img = img[y:y+h, x:x+w]
    print(os.path.basename(data),img.shape)
    # 保存图像
    save_path = os.path.join(save_dir,os.path.basename(data))
    cv2.imwrite(save_path,img)
    return img

def main():
    data_path = '/braindat/large-scale-EM-data/Kasthuri2015/Thousands_6nm_spec_lossless'
    save_dir = '/braindat/lab/chenyd/DATASET/unlabel_data/Kasthuri2015_png'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data_list,data_len = get_data_list(data_path)
    print(data_len)
    reference_img = cv2.imread(data_list[0],0)
    _,x,y,w,h = remove_black_border(reference_img)
    print(x,y,w,h)
    # referce_img = cv2.imread(data_list[0],0)
    # referce_img, M = remove_black_border_and_correct(referce_img)
    # print(referce_img.shape ,'lets start')
    # data_list = data_list[:10]
    # for data in data_list:
    #     process(data)
    #     print(data)
    #     break
    # print('done')
    cpu_num = os.cpu_count()
    print('cpu_num:',cpu_num)
    pool = Pool(processes=cpu_num-8)
    for i in data_list:
        pool.apply_async(process, args=(i,save_dir,x,y,w,h))
    pool.close()
    pool.join()
    print('done')

if __name__ == '__main__':
    main()