'''
@File     : get_pic.py
@Copyright:
@author   : lxt
@Date     : 2019/5/15
@Desc     :
'''

#  实例分割
#  获取图片和json文件
import cv2 as cv
import random
import glob
import os
from PIL import Image
import shutil

def get_samples(foldername, savePath):
    print('savePath:', savePath)
    if os.path.exists(savePath) is False:
        os.makedirs(savePath)

    filenames = os.listdir(foldername)

    for filename in filenames:
        full_path = os.path.join(foldername, filename)
        # print(filename[:-5])
        new_name = filename[:-5] + '.png'
        label_png = os.listdir(full_path)[3]
        # print(os.listdir(full_path))
        # os.rename(os.path.join(filename, label_png),os.path.join(filename, name))
        shutil.copy(os.path.join(full_path, label_png), os.path.join(savePath, label_png))
        os.rename(os.path.join(savePath, label_png), os.path.join(savePath, new_name))
        # print(os.listdir(filename))

if __name__ == '__main__':
    savePath = './segmentation/train_data/cv2_mask'
    get_samples('./segmentation/train_data/labelme_json', savePath)