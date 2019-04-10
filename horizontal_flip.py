import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import os
import matplotlib.patches as patches

from lxml.etree import Element, SubElement, tostring
from xml.dom.minidom import parseString
import xml.etree.ElementTree as ET
import numpy as np

def showimg(img):
    channelNum = len(img.shape)

    if channelNum == 3:
        fig = plt.subplots(1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if channelNum == 2:
        fig = plt.subplots(1), plt.imshow(img)


def scaleimg(img, scale=1.0):
    H, W, C = img.shape
    size = (int(scale * W), int(scale * H))
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    del H, W, C, size, scale
    return img.copy()


# img = rotateimg(image, angle)
def rotateimg(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)  # 给的角度为正的时候，则逆时针旋转
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
    #    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated  # 返回旋转后的图像, angle是角度制，不是弧度制


'''
1， 读取xml返回结果
    输入：CLASS_NAMES元组   xml路径
    返回：(H, W, boxes)   boxes是一个二维np数组，6列分别为
          id classid xmin xmax ymin ymax
           0       1    2    3    4    5

2,  将boxes  CLASS_NAMES  H W 信息写入xml
    输入：boxes  CLASS_NAMES  H W   xml路径
    输出：硬盘上的一个xml

3， 根据 img，boxes数组，class_names，画出一个图来
    输入：img，boxes，class_names
'''



# CLASS_NAMES = ('person', 'dog')  # 下标从0开始，这里可以没有顺序，最好有顺序

#          id classid xmin xmax ymin ymax
#           0       1    2    3    4    5
def xml2boxes(xmlpath, CLASS_NAMES):
    print("xmlpath:", xmlpath)

    cls_to_idx = dict(zip(CLASS_NAMES, range(len(CLASS_NAMES))))
    idx_to_cls = dict(zip(range(len(CLASS_NAMES)), CLASS_NAMES))

    #    print(cls_to_idx)
    #    print(idx_to_cls)

    annotations = ET.parse(xmlpath)
    # 获得 HWC
    size = annotations.find('size')
    W = int(size.find('width').text)
    H = int(size.find('height').text)
    C = int(size.find('depth').text)
    # 获得类别和具体坐标
    bbox = list()
    count = 1
    for obj in annotations.iter('object'):  # 提取 xml文件中的信息
        line = []
        bndbox_anno = obj.find('bndbox')
        # xmin等从 1 开始计数
        tmp = map(int, [bndbox_anno.find('xmin').text,
                        bndbox_anno.find('xmax').text,
                        bndbox_anno.find('ymin').text,
                        bndbox_anno.find('ymax').text])
        tmp = list(tmp)  # 1 x 4

        name = obj.find('name').text.lower().strip()

        line.append(count)
        line.append(cls_to_idx[name])
        line.append(tmp[0])
        line.append(tmp[1])
        line.append(tmp[2])
        line.append(tmp[3])
        count = count + 1
        #        print(line)
        bbox.append(line)

    boxes = np.stack(bbox).astype(np.int32)
    return boxes, H, W


# boxes, H, W = xml2boxes("1.xml", CLASS_NAMES)
# print("boxes:\n", boxes)
# 对只有一个类别的时候，CLASS_NAMES要在后面加一个字符串
# 比如 CLASS_NAMES = ("apple", "xxxx") 这是个bug，还没修

######################################################
# boxes2xml_labelImg(boxes, CLASS_NAMES, H, W, xmlpath, wrtin_img_folder_name, imgName, img_fullpath)
def boxes2xml_labelImg(boxes, CLASS_NAMES, H, W, xmlpath, wrtin_img_folder_name,
                       imgName, img_fullpath):
    '''
    这是一个labelImg可以查看的版本

    这个时候要求 CLASS_NAMES 是有顺序的，和boxes里头的第二列
    的类别id要一一对应
    '''
    cls_to_idx = dict(zip(CLASS_NAMES, range(len(CLASS_NAMES))))
    idx_to_cls = dict(zip(range(len(CLASS_NAMES)), CLASS_NAMES))

    node_annotation = Element('annotation')
    #################################################
    node_folder = SubElement(node_annotation, 'folder')
    node_filename = SubElement(node_annotation, 'filename')
    node_path = SubElement(node_annotation, 'path')

    node_source = SubElement(node_annotation, 'source')
    node_database = SubElement(node_source, 'database')

    node_folder.text = wrtin_img_folder_name  # 这个是定死的，赋值一次就不会变了
    node_filename.text = imgName  # 图片的文件名，不包含后缀
    node_path.text = img_fullpath  # 随着文件名变化

    node_database.text = "Unknown"

    node_size = SubElement(node_annotation, 'size')
    #################################################
    # node_size
    node_width = SubElement(node_size, 'width')
    node_height = SubElement(node_size, 'height')
    node_depth = SubElement(node_size, 'depth')

    node_width.text = str(W)
    node_height.text = str(H)
    node_depth.text = str(3)  # 默认是彩色
    #################################################
    node_segmented = SubElement(node_annotation, 'segmented')
    node_segmented.text = "0"
    #################################################

    # node_object  若干    要循环
    for i in range(boxes.shape[0]):
        node_object = SubElement(node_annotation, 'object')
        classid = boxes[i, 1]
        #    print(idx_to_cls[classid])
        node_name = SubElement(node_object, 'name')
        node_name.text = idx_to_cls[classid]

        node_pose = SubElement(node_object, 'pose')
        node_truncated = SubElement(node_object, 'truncated')
        node_Difficult = SubElement(node_object, 'Difficult')

        node_pose.text = "Unspecified"
        node_truncated.text = "1"
        node_Difficult.text = "0"

        node_bndbox = SubElement(node_object, 'bndbox')

        node_xmin = SubElement(node_bndbox, 'xmin')
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_ymax = SubElement(node_bndbox, 'ymax')

        node_xmin.text = str(boxes[i, 2])
        node_xmax.text = str(boxes[i, 3])
        node_ymin.text = str(boxes[i, 4])
        node_ymax.text = str(boxes[i, 5])

    ###################
    xml = tostring(node_annotation, pretty_print=True)  # 格式化显示，该换行的换行
    dom = parseString(xml)

    test_string = xml.decode('utf-8')
    # print('test:\n', test_string)

    with open(xmlpath, "w") as text_file:
        text_file.write(test_string)


######################################################
def drawboxes(imgpath, boxes, CLASS_NAMES):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import cv2

    cls_to_idx = dict(zip(CLASS_NAMES, range(len(CLASS_NAMES))))
    idx_to_cls = dict(zip(range(len(CLASS_NAMES)), CLASS_NAMES))

    if isinstance(imgpath, str):
        img = cv2.imread(imgpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if isinstance(imgpath, np.ndarray):
        img = imgpath

    fig, ax = plt.subplots(1)

    for i in range(boxes.shape[0]):
        bndbox = list(boxes[i, :])
        x = bndbox[2]
        y = bndbox[4]
        w = bndbox[3] - bndbox[2]
        h = bndbox[5] - bndbox[4]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)
        name = idx_to_cls[boxes[i, 1]]
        ax.text(x - 5, y - 5, name, style='italic', color='yellow', fontsize=12)

    ax.imshow(img)


# drawboxes("1.jpg", boxes, CLASS_NAMES)

##################################


def getFilenames(filepath):
    '''得到一个文件夹下所有的文件名，不包含后缀
    '''
    filelist = os.listdir(filepath)

    filenames = []

    for files in filelist:
        filename = os.path.splitext(files)[0]
        #    print(files)
        #    print(filename)
        filenames.append(filename)
    return filenames


def fliplr_boxes(boxes, W):
    ''' 对boxes做水平翻转'''
    boxes_copy = boxes.copy()

    xmin = boxes[:, 2].copy()
    xmax = boxes[:, 3].copy()
    boxes_copy[:, 3] = W - 1 - xmin  # 注意这里不是 2,3 是 3,2 不然xmin会大于xmax
    boxes_copy[:, 2] = W - 1 - xmax
    return boxes_copy


def main():
    img_read_path = "./origin data/origin/imgs"
    xml_read_path = "./origin data/origin/xml"

    img_write_path = "./origin data/lr_flip"  # 图片和xml水平翻转后的写入文件夹
    xml_write_path = "./origin data/lr_flip"

    if not os.path.exists(img_write_path):
        os.makedirs(img_write_path)

    filenames = getFilenames(xml_read_path)

    CLASS_NAMES = ('door flame', 'person', 'aa')  # 这里有个bug懒得改，一个类别时也要写两个进去

    count = 8000

    wrtin_img_folder_name = "fliped"

    # for i in range(len(filenames)):
    #     name = filenames[i]

    for f in os.listdir(img_read_path):
        name = os.path.splitext(f)

        imgname = img_read_path + "/" + f
        img = cv2.imread(imgname)

        xmlname = xml_read_path + "/" + str(name[0]) + ".xml"
        boxes, H, W = xml2boxes(xmlname, CLASS_NAMES)
        #    print("xmlname:", xmlname)

        H, W, C = img.shape
        ##############################
        fliped_boxes = fliplr_boxes(boxes, W)
        fliped_img = cv2.flip(img, 1)
        ##############################
        FileName = str(count)

        jpgpath = img_write_path + "/" + FileName + ".jpg"
        cv2.imwrite(jpgpath, fliped_img)

        xmlpath = xml_write_path + "/" + FileName + ".xml"
        boxes2xml_labelImg(fliped_boxes, CLASS_NAMES, H, W, xmlpath, wrtin_img_folder_name, FileName, jpgpath)

        count = count + 1
    print("completed", count)

if __name__ == '__main__':
    main()

