from imgaug import augmenters as iaa
import cv2
import os
from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import PIL.Image as Image
import pathlib2
import shutil
import imageio
import xml.etree.ElementTree as ET
import numpy as np
import random



# imgaug test
# iaa.ContrastNormalization((0.5, 2.0)),
seq = iaa.Sequential([
    iaa.Multiply((1.2, 1.4))
    # iaa.ContrastNormalization((0.5, 2.0))
])

CDIR = 'contrast'
BDIR = 'brightness'
FDIR = 'fliped'

if not os.path.exists('./origin data/contrast/'):
    os.makedirs('./origin data/contrast/')

if not os.path.exists('./origin data/brightness/'):
    os.makedirs('./origin data/brightness/')


# # 1 生成对比度增强数据
# root_path = 'F:\\pyCase\\data_augmentation'
# img_read_path='./origin data/origin/imgs/'
# xml_read_path = './origin data/origin/xml/'
# contrast_path='./origin data/contrast/'
# save_xml_path = './origin data/contrast/'
#

# ni = 8509
# for i in os.listdir(img_read_path):
#     ni += 1
#     a, b = os.path.splitext(i)
#     image = imageio.imread(img_read_path+i)
#     image_aug_list = seq.augment_images(image)
#     imageio.imwrite(contrast_path+str(ni)+'.jpg', image_aug_list)
#     xmlname = xml_read_path + a + ".xml"
#     save_path = save_xml_path + str(ni) + '.xml'
#     tree = ET.parse(xmlname)
#     root = tree.getroot()
#     folder1 = root.find('folder')
#     folder1.text = CDIR
#     filename = root.find('filename')
#     filename.text = str(ni)+'.jpg'
#     path = root.find('path')
#     path.text = root_path + contrast_path + "\\" + str(ni)+".jpg"
#     tree.write(save_path)
# print("completed")



# # 1 生成亮度增强数据
root_path = 'F:\\pyCase\\data_augmentation'
img_read_path='./origin data/origin/imgs/'
xml_read_path = './origin data/origin/xml/'
brightness_path='./origin data/brightness/'
save_xml_path = './origin data/brightness/'

ni = 9019
for i in os.listdir(img_read_path):
    ni += 1
    a, b = os.path.splitext(i)
    image = imageio.imread(img_read_path+i)
    image_aug_list = seq.augment_images(image)
    imageio.imwrite(brightness_path+str(ni)+'.jpg', image_aug_list)
    xmlname = xml_read_path + a + ".xml"
    save_path = save_xml_path + str(ni) + '.xml'
    tree = ET.parse(xmlname)
    root = tree.getroot()
    folder1 = root.find('folder')
    folder1.text = BDIR
    filename = root.find('filename')
    filename.text = str(ni)+'.jpg'
    path = root.find('path')
    path.text = root_path + brightness_path + "\\" + str(ni)+".jpg"
    tree.write(save_path)
print("completed")

