'''
@File     : brightness_contrast.py
@Copyright:
@author   : lxt
@Date     : 2019/3/15
@Desc     :
'''
# 对比度和亮度变化

from imgaug import augmenters as iaa
import imageio
import xml.etree.ElementTree as ET
import os
import argparse

parser = argparse.ArgumentParser(description = 'this is a description')
parser.add_argument('--choose', '-c', required  = True, type = int)
parser.add_argument('--start', '-s', required  = True, type = int)
# 将变量以标签-值的字典形式存入args字典
args = parser.parse_args()

if not os.path.exists('./detection_data/contrast/'):
    os.makedirs('./detection_data/contrast/')

if not os.path.exists('./detection_data/brightness/'):
    os.makedirs('./detection_data/brightness/')

CDIR = 'contrast'
BDIR = 'brightness'
FDIR = 'fliped'
root_path = 'F:\\pyCase\\data_augmentation'
img_read_path='./detection_data/origin/imgs/'
xml_read_path = './detection_data/origin/xml/'

# # 2 生成对比度增强数据
def contrast_aug(ni = 8509):
    contrast_path='./detection_data/contrast/'
    save_xml_path = './detection_data/contrast/'

    for i in os.listdir(img_read_path):
        ni += 1
        a, b = os.path.splitext(i)
        image = imageio.imread(img_read_path+i)
        image_aug_list = seq.augment_images(image)
        imageio.imwrite(contrast_path+str(ni)+'.jpg', image_aug_list)
        xmlname = xml_read_path + a + ".xml"
        save_path = save_xml_path + str(ni) + '.xml'
        tree = ET.parse(xmlname)
        root = tree.getroot()
        folder1 = root.find('folder')
        folder1.text = CDIR
        filename = root.find('filename')
        filename.text = str(ni)+'.jpg'
        path = root.find('path')
        path.text = root_path + contrast_path + "\\" + str(ni)+".jpg"
        tree.write(save_path)
    print("completed")


# # 1 生成亮度增强数据
def brightness_aug(ni = 9019):
    brightness_path='./detection_data/brightness/'
    save_xml_path = './detection_data/brightness/'

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

if __name__ == '__main__':
    choose = args.choose
    start_ids = args.start
    # choose为1，表示亮度增强
    if choose == 1:
        seq = iaa.Sequential([
            iaa.Multiply((1.2, 1.4))
        ])
        brightness_aug(start_ids)
    # choose为2，表示对比度增强
    elif choose == 2:
        seq = iaa.Sequential([
            iaa.ContrastNormalization((0.5, 2.0))
        ])
        contrast_aug(start_ids)
