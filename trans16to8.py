'''
@File     : trans16to8.py
@Copyright:
@author   : lxt
@Date     : 2019/5/15
@Desc     :
'''
# 实例分割
# 用于把16位图转为8位图

from PIL import Image
import numpy as np
import os
n=510#n为.json文件个数
dirpath = './segmentation/datasets/'
# 生成8bit的label.png
# for i in range(n):
#     open_path = dirpath + str(i)+'_json'+'/label.png'#文件地址
#     print(os.path.exists(open_path))
#     img1=Image.open(open_path)#打开图像
#     save_path=dirpath + str(i)+'_json'+'/label_8.png'  #保存地址
#     print(save_path)
#     img=Image.fromarray(np.uint8(img1))#16位转换成8位
#     img.save(save_path) #保存成png格式

# 可视
for i in range(n):
    open_path = dirpath + str(i)+'_json'+'/label.png'#文件地址
    print(os.path.exists(open_path))
    img1=Image.open(open_path)#打开图像
    save_path=dirpath + str(i)+'_json'+'/label_8_via.png'  #保存地址
    print(save_path)
    img=Image.fromarray(np.uint8(img1)*30)#16位转换成8位  提高类别之间的差异性
    img.save(save_path) #保存成png格式