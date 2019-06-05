'''
@File     : random_crop.py
@Copyright:
@author   : lxt
@Date     : 2019/3/16
@Desc     :
'''

# 用于随机裁剪图片

import cv2
import os
import random
import voc_xml
import utils
from voc_xml import CreateXML


def crop_img(src, top_left_x, top_left_y, crop_w, crop_h):
    '''裁剪图像    '''
    rows, cols, n_channel = src.shape
    row_min, col_min = int(top_left_y), int(top_left_x)
    row_max, col_max = int(row_min + crop_h), int(col_min + crop_w)
    if row_max > rows or col_max > cols:
        print("crop size err: src->%dx%d,crop->top_left(%d,%d) %dx%d" % (
        cols, rows, col_min, row_min, int(crop_w), int(crop_h)))
        return None
    # 图像矩阵裁剪
    crop_img = src[row_min:row_max, col_min:col_max]
    return crop_img


def crop_xy(x, y, top_left_x, top_left_y, crop_w, crop_h):
    ''' 坐标平移变换    '''
    crop_x = int(x - top_left_x)
    crop_y = int(y - top_left_y)
    crop_x = utils.confine(crop_x, 0, crop_w - 1)
    crop_y = utils.confine(crop_y, 0, crop_h - 1)
    return crop_x, crop_y


def crop_box(box, top_left_x, top_left_y, crop_w, crop_h, iou_thr=0.5):
    '''目标框坐标平移变换'''
    xmin, ymin = crop_xy(box[0], box[1], top_left_x, top_left_y, crop_w, crop_h)
    xmax, ymax = crop_xy(box[2], box[3], top_left_x, top_left_y, crop_w, crop_h)
    croped_box = [xmin, ymin, xmax, ymax]
    if utils.calc_iou([0, 0, box[2] - box[0], box[3] - box[1]], [0, 0, xmax - xmin, ymax - ymin]) < iou_thr:
        croped_box = [0, 0, 0, 0]
    return croped_box


def crop_xml(crop_img_name, xml_tree, top_left_x, top_left_y, crop_w, crop_h, iou_thr=0.5):
    '''xml目标框裁剪变换'''
    root = xml_tree.getroot()
    size = root.find('size')
    depth = int(size.find('depth').text)
    createdxml = CreateXML(crop_img_name, int(crop_w), int(crop_h), depth)
    for obj in root.iter('object'):
        obj_name = obj.find('name').text
        xml_box = obj.find('bndbox')
        xmin = int(xml_box.find('xmin').text)
        ymin = int(xml_box.find('ymin').text)
        xmax = int(xml_box.find('xmax').text)
        ymax = int(xml_box.find('ymax').text)
        box = crop_box([xmin, ymin, xmax, ymax], top_left_x, top_left_y, crop_w, crop_h, iou_thr)
        if (box[0] >= box[2]) or (box[1] >= box[3]):
            continue
        createdxml.add_object_node(obj_name, box[0], box[1], box[2], box[3])
    return createdxml


def crop_img_and_xml(img, xml_tree, crop_img_name, top_left_x, top_left_y, crop_w, crop_h, iou_thr):
    '''裁剪图像和xml目标框'''
    croped_img = crop_img(img, top_left_x, top_left_y, crop_w, crop_h)
    if croped_img is None:
        return None
    croped_xml = crop_xml(crop_img_name, xml_tree, top_left_x, top_left_y, crop_w, crop_h, iou_thr)
    return croped_img, croped_xml


def random_crop(imgs_dir, xmls_dir, imgs_save_dir, xmls_save_dir, crop_type='RANDOM_CROP', crop_n=1, dsize=(0, 0), fw=1.0, fh=1.0, random_wh=False, iou_thr=0.5):
    '''随机裁剪指定路径下的图片和xml '''

    for file in os.listdir(imgs_dir):
        print(file)
        name = os.path.splitext(file)
        img_file = os.path.join(imgs_dir, file)
        xml_file = os.path.join(xmls_dir, name[0] + '.xml')

        img = cv2.imread(img_file)
        imgh, imgw, n_channels = img.shape

        if crop_type == 'CENTER_CROP':
            crop_n = 1
        elif crop_type == 'FIVE_CROP':
            crop_n = 5

        for i in range(crop_n):
            crop_imgw, crop_imgh = dsize
            if dsize == (0, 0) and not random_wh:
                crop_imgw = int(imgw * fw)
                crop_imgh = int(imgh * fh)
            elif random_wh:
                crop_imgw = int(imgw * (fw + random.random() * (1 - fw)))
                crop_imgh = int(imgh * (fh + random.random() * (1 - fh)))

            if crop_type == 'RANDOM_CROP':
                crop_top_left_x, crop_top_left_y = random.randint(0, imgw - crop_imgw - 1), random.randint(0, imgh - crop_imgh - 1)

            elif crop_type == 'CENTER_CROP':
                crop_top_left_x, crop_top_left_y = int(imgw / 2 - crop_imgw / 2), int(imgh / 2 - crop_imgh / 2)
            elif crop_type == 'FIVE_CROP':
                if i == 0:
                    crop_top_left_x, crop_top_left_y = 0, 0
                elif i == 1:
                    crop_top_left_x, crop_top_left_y = imgw - crop_imgw - 1, 0
                elif i == 2:
                    crop_top_left_x, crop_top_left_y = 0, imgh - crop_imgh - 1
                elif i == 3:
                    crop_top_left_x, crop_top_left_y = imgw - crop_imgw - 1, imgh - crop_imgh - 1
                else:
                    crop_top_left_x, crop_top_left_y = int(imgw / 2 - crop_imgw / 2), int(imgh / 2 - crop_imgh / 2)
            else:
                print('crop type wrong! expect [RANDOM_CROP,CENTER_CROP,FIVE_CROP]')

            croped_img_name = name[0] + '_' + str(crop_top_left_x) + '_' + str(crop_top_left_y) + '_wh' + str(crop_imgw) + 'x' + str(crop_imgh) + name[1]
            # 调用剪切和xml文件
            croped = crop_img_and_xml(img, voc_xml.get_xml_tree(xml_file), croped_img_name, crop_top_left_x,
                                  crop_top_left_y, crop_imgw, crop_imgh, iou_thr)
            imgcrop, xmlcrop = croped[0], croped[1]
            cv2.imwrite(os.path.join(imgs_save_dir, croped_img_name), imgcrop)
            xmlcrop.save_xml(xmls_save_dir, croped_img_name.split('.')[0] + '.xml')



def main():
    imgs_dir = './detection_data/forth/imgs2/'
    xmls_dir = './detection_data/forth/xml2/'
    imgs_save_dir = './detection_data/forth/imgs2/'
    if not os.path.exists(imgs_save_dir):
        os.makedirs(imgs_save_dir)
    xmls_save_dir = './detection_data/forth/imgs2/'
    if not os.path.exists(xmls_save_dir):
        os.makedirs(xmls_save_dir)

    crop_type = 'RANDOM_CROP'  # ['RANDOM_CROP','CENTER_CROP','FIVE_CROP']
    crop_n = 2 # 每张原图 crop 5张图
    dsize = (400, 300)  # 指定裁剪尺度
    fw = 0.4
    fh = 0.4  # 指定裁剪尺度比例
    random_wh = True  # 是否随机尺度裁剪，若为True,则dsize指定的尺度失效
    iou_thr = 0.25  # 裁剪后目标框大小与原框大小的iou值大于该阈值则保留
    # print(type(crop_n))
    random_crop(imgs_dir, xmls_dir, imgs_save_dir, xmls_save_dir, crop_type, crop_n, dsize, fw, fh, random_wh, iou_thr)
    print("completed")


if __name__ == '__main__':
    main()
