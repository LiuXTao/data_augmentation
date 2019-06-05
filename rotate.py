'''
@File     : rotate.py
@Copyright:
@author   : lxt
@Date     : 2019/3/15
@Desc     :
'''

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import math

def rotate_image(src, angle, scale=1):
    w = src.shape[1]
    h = src.shape[0]
    # 角度变弧度
    rangle = np.deg2rad(angle)  # angle in radians
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    dst = cv2.warpAffine(src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
    # 仿射变换
    return dst

# 对应修改xml文件
def rotate_xml(src, xmin, ymin, xmax, ymax, angle, scale=1.):
    w = src.shape[1]
    h = src.shape[0]
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    # 获取旋转后图像的长和宽
    nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
    nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)  # 第一个参数表示center坐标
    # calculate the move from the old center to the new center combined
    # with the rotation
    # 计算从旧中心到旋转后的新中心的步长
    rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5, 0]))
    # the move only affects the translation, so update the translation part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    # rot_mat是最终的旋转矩阵
    point1 = np.dot(rot_mat, np.array([xmin, ymin, 1]))             #这种新画出的框大一圈
    point2 = np.dot(rot_mat, np.array([xmax, ymin, 1]))
    point3 = np.dot(rot_mat, np.array([xmax, ymax, 1]))
    point4 = np.dot(rot_mat, np.array([xmin, ymax, 1]))
    # point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))   # 获取原始矩形的四个中点，然后将这四个点转换到旋转后的坐标系下
    # point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
    # point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
    # point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))
    concat = np.vstack((point1, point2, point3, point4))            # 合并np.array
    # 改变array类型
    concat = concat.astype(np.int32)
    rx, ry, rw, rh = cv2.boundingRect(concat)   #rx,ry,为新的外接框左上角坐标，rw为框宽度，rh为高度，新的xmax=rx+rw,新的ymax=ry+rh
    return rx, ry, rw, rh


def main():
    xmlpath = './detection_data/origin/xml/'     # 源图像路径
    imgpath = './detection_data/origin/imgs/'    # 源图像所对应的xml文件路径
    rotated_imgpath = './detection_data/rotate/'
    rotated_xmlpath = './detection_data/rotate/'

    if not os.path.exists(rotated_imgpath):
        os.makedirs(rotated_imgpath)

    for angle in (30, 90, 270, 330):
        # if angle != 90:
        #     continue
        if not os.path.exists(rotated_imgpath+'/'+str(angle)):
            os.makedirs(rotated_imgpath+'/'+str(angle))

        for i in os.listdir(imgpath):
            a, b = os.path.splitext(i)  # 分离出文件名a
            img = cv2.imread(imgpath + i)
            # 旋转图片
            rotated_img = rotate_image(img, angle)
            cv2.imwrite(rotated_imgpath + '/' + str(angle) + '/' + a + '.jpg', rotated_img)

            tree = ET.parse(xmlpath + a + '.xml')
            root = tree.getroot()

            w = img.shape[1]
            h = img.shape[0]
            rangle = np.deg2rad(angle)
            nw = int(abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w))
            nh = int(abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w))
            size = root.find('size')
            size.find('width').text = str(nw)
            size.find('height').text = str(nh)

            for box in root.iter('bndbox'):
                xmin = float(box.find('xmin').text)
                ymin = float(box.find('ymin').text)
                xmax = float(box.find('xmax').text)
                ymax = float(box.find('ymax').text)
                # 计算旋转后的box
                x, y, w, h = rotate_xml(img, xmin, ymin, xmax, ymax, angle)
                box.find('xmin').text = str(x)
                box.find('ymin').text = str(y)
                box.find('xmax').text = str(x + w)
                box.find('ymax').text = str(y + h)
            tree.write(rotated_xmlpath + '/' + str(angle) + '/' + a + '.xml')
            # print(str(a) + '.xml has been rotated for ' + str(angle) + '°')

if __name__ == '__main__':
    main()

