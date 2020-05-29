#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time    :2020/5/21 00:26
# @Author  : Cathy 
# @FileName: detection_demo.py


import os
import time
import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision
from PIL import Image
from matplotlib import pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# classes_BDD
BDD_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'bus', 'traffic light', 'traffic sign',
    'person', 'bike', 'truck', 'motor', 'car', 'train', 'rider'
]


if __name__ == "__main__":

    base_path = './data/bdd100k/images/100k/test/'
    img_paths = os.listdir(base_path)[:3]

    for i in img_paths:
        path_img = base_path + i

        # config
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])

        # 1. load data & model
        input_image = Image.open(path_img).convert("RGB")
        # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model = torch.load('./models/2_model.pth.tar', map_location='cpu')
        model.eval()

        # 2. preprocess
        img_chw = preprocess(input_image)

        # 3. to device
        if torch.cuda.is_available():
            img_chw = img_chw.to('cuda')
            model.to('cuda')

        # 4. forward
        input_list = [img_chw]
        with torch.no_grad():
            tic = time.time()
            print("input img tensor shape:{}".format(input_list[0].shape))
            output_list = model(input_list)
            output_dict = output_list[0]
            print("pass: {:.3f}s".format(time.time() - tic))
            for k, v in output_dict.items():
                print("key:{}, value:{}".format(k, v))

        # 5. visualization
        out_boxes = output_dict["boxes"].cpu()
        out_scores = output_dict["scores"].cpu()
        out_labels = output_dict["labels"].cpu()

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(input_image, aspect='equal')

        num_boxes = out_boxes.shape[0]
        max_vis = 40
        thres = 0.5

        for idx in range(0, min(num_boxes, max_vis)):

            score = out_scores[idx].numpy()
            bbox = out_boxes[idx].numpy()
            class_name = BDD_INSTANCE_CATEGORY_NAMES[out_labels[idx]]

            if score < thres:
                continue

            ax.add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False,
                                       edgecolor='red', linewidth=3.5))
            ax.text(bbox[0], bbox[1] - 2, '{:s} {:.3f}'.format(class_name, score), bbox=dict(facecolor='blue', alpha=0.5),
                    fontsize=14, color='white')
        plt.title(i)
        plt.savefig('./det_images/epoch2/' + i)
        plt.show()
        plt.close()

