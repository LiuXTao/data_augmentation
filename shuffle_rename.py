'''
@File     : shuffle_rename.py
@Copyright:
@author   : lxt
@Date     : 2019/3/18
@Desc     :
'''

import os
import xml.etree.ElementTree as ET
import random
import shutil
import json

def shufflelist(imgList):
    random.shuffle(imgList)
    no = 0
    for i in imgList:
        li = os.path.splitext(i)
        print(li)
        os.rename(os.path.join(rootDir, i), os.path.join(rootDir, str(no) + li[-1]))
        os.rename(os.path.join(rootJ, li[0].strip() + '.json'), os.path.join(rootJ, str(no) +'.json'))
        # print(os.path.join(rootDir, str(no) + li[-1]))
        # print(os.path.join(rootJ, li[0] + '.json'))
        # print(os.path.exists('./segmentation/json/' + li[0].strip() + '.json'))
        # print(os.path.exists(os.path.join(rootDir, i)))
        no += 1
    print(no)

def rename(jsonDir):
    for i in os.listdir(os.path.abspath(jsonDir)):
        # print(i)
        li = os.path.splitext(i)
        print(li)
        filepath = os.path.join(rootJ, i)
        after = None

        with open(filepath, 'rb') as f:
            data = json.load(f)
            # print(type(data))
            old_name = data["imagePath"]
            # print(old_name)
            # print(old_name.split('.')[-1])
            new_name = li[0] + '.' +old_name.split('.')[-1]
            # print(new_name)
            after = data
            after["imagePath"] = new_name
            print(type(after))
            # print(after["imagePath"] )

        with open(filepath, 'w') as f:
            data = json.dump(after, f)

def rename2(jsonDir):
    for i in os.listdir(os.path.abspath(jsonDir)):
        # print(i)
        li = os.path.splitext(i)
        print(li[0])
        filepath = os.path.join(rootJ, i)
        with open(filepath, 'rb') as f:
            data = json.load(f)
            # print(type(data))
            old_name = data["imagePath"]
            la = old_name.split('.')
        imgpath = os.path.join(rootDir, li[0]+'.'+la[-1])
        # print(os.path.exists(imgpath))
        # print(old_name)
        os.rename(imgpath, os.path.join(rootDir, old_name))
        os.rename(filepath, os.path.join(rootJ, la[0] +'.json'))

if __name__ == '__main__':
    fileDir = './segmentation/new_imgs/'
    jsonDir = './segmentation/new_json/'
    rootDir = os.path.abspath(fileDir)
    rootJ = os.path.abspath(jsonDir)
    imgList = os.listdir(os.path.abspath(fileDir))

    rename2(jsonDir)
