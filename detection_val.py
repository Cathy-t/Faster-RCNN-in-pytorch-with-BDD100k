#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time    :2020/5/20 13:42
# @Author  : Cathy 
# @FileName: detection_val.py

import os
import time
import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# classes_BDD
BDD_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'bus', 'traffic light', 'traffic sign',
    'person', 'bike', 'truck', 'motor', 'car', 'train', 'rider'
]

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, time):
            return obj.__str__()
        else:
            return super(NpEncoder, self).default(obj)


if __name__ == "__main__":

    # # self
    model = torch.load('./models/3_model.pth.tar')

    # # origal
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()

    vis_dir = os.path.join(BASE_DIR, "data", "bdd100k", "images", '100k', 'val')
    img_names = list(filter(lambda x: x.endswith(".jpg"), os.listdir(vis_dir)))
    preprocess = transforms.Compose([transforms.ToTensor(), ])

    jsontexts = list()

    for i in range(0, len(img_names)):

        path_img = os.path.join(vis_dir, img_names[i])
        # preprocess
        input_image = Image.open(path_img).convert("RGB")
        img_chw = preprocess(input_image)

        # to device
        if torch.cuda.is_available():
            img_chw = img_chw.to('cuda')
            model.to('cuda')

        # forward
        input_list = [img_chw]
        with torch.no_grad():
            tic = time.time()
            output_list = model(input_list)
            output_dict = output_list[0]

            # result to json
            out_boxes = output_dict["boxes"].cpu()
            out_scores = output_dict["scores"].cpu()
            out_labels = output_dict["labels"].cpu()

            # 确定最终输出的超参
            num_boxes = out_boxes.shape[0]
            max_vis = num_boxes
            thres = 0.5

            for idx in range(0, min(num_boxes, max_vis)):
                score = out_scores[idx].numpy()
                bbox = out_boxes[idx].numpy()
                class_name = BDD_INSTANCE_CATEGORY_NAMES[out_labels[idx]]

                if score < thres:
                    continue

                jsontext = {
                        'name': path_img.split('\\')[-1].split('.')[0],
                        'timestamp': 1000,
                        'category': class_name,
                        'bbox': bbox,
                        'score': score
                    }

                jsontexts.append(jsontext)

    print("pass: {:.3f}s".format(time.time() - tic))

    json_str = json.dumps(jsontexts, indent=4, cls=MyEncoder)
    with open('./result/val_result_3.json', 'w') as json_file:
        json_file.write(json_str)

    print('Done!!!!')
