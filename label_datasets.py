'''
@File     : label_datasets.py
@Copyright:
@author   : lxt
@Date     : 2019/3/15
@Desc     :
'''
# 实例分割用

import argparse
import base64
import json
import os
import os.path as osp
import warnings

import PIL.Image
import yaml

from labelme import utils

def main():
    warnings.warn("This script is aimed to demonstrate how to convert the\n"
                  "JSON file to a single image dataset, and not to handle\n"
                  "multiple JSON files to generate a real-use dataset.")

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--json_file', default='./segmentation/json/')
    parser.add_argument('-o', '--out', default='./segmentation/datasets')
    args = parser.parse_args()

    json_file = args.json_file

    list_file = os.listdir(json_file)

    #     if args.out is None:
    #         out_dir = osp.basename(json_file).replace('.', '_')
    #         out_dir = osp.join(osp.dirname(json_file), out_dir)
    #     else:
    #         out_dir = args.out
    #     if not osp.exists(out_dir):
    #         os.mkdir(out_dir)

    for i in range(0, len(list_file)):
        # if args.out is None:
        out_dir = osp.basename(list_file[i]).replace('.', '_')
        out_dir = osp.join(args.out, out_dir)
        # else:
        #     out_dir = args.out
        if not osp.exists(out_dir):
            os.mkdir(out_dir)

        path = os.path.join(json_file, list_file[i])
        data = json.load(open(path))

        if data['imageData']:
            imageData = data['imageData']
        else:
            imagePath = os.path.join(os.path.dirname(json_file), data['imagePath'])
            with open(imagePath, 'rb') as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode('utf-8')
        img = utils.img_b64_to_arr(imageData)

        label_name_to_value = {'_background_': 0}
        for shape in sorted(data['shapes'], key=lambda x: x['label']):
            label_name = shape['label']
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value
        lbl = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

        label_names = [None] * (max(label_name_to_value.values()) + 1)
        for name, value in label_name_to_value.items():
            label_names[value] = name
        lbl_viz = utils.draw_label(lbl, img, label_names)

        PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
        utils.lblsave(osp.join(out_dir, 'label.png'), lbl)
        PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))

        with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
            for lbl_name in label_names:
                f.write(lbl_name + '\n')

        warnings.warn('info.yaml is being replaced by label_names.txt')
        info = dict(label_names=label_names)
        with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
            yaml.safe_dump(info, f, default_flow_style=False)

        print('Saved to: %s' % out_dir)


if __name__ == '__main__':
    main()