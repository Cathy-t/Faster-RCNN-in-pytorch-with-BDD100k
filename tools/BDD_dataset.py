#!/usr/bin/env python
# _*_coding:utf-8 _*_
# @Time    :2020/5/17 14:56
# @Author  : Cathy 
# @FileName: detection_demo.py

import numpy as np
import torch
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import json

random.seed(1)
rmb_label = {"1": 0, "100": 1}


class bddDataset(object):
    def __init__(self, data_dir, transforms, flag, label_list):

        self.data_dir = data_dir
        self.transforms = transforms
        if flag == 'train':
            self.img_dir = os.path.join(data_dir, "images", '100k', 'train')
            self.json_dir = os.path.join(data_dir, "labels", 'bdd100k_labels_images_train.json')
        if flag == 'val':
            self.img_dir = os.path.join(data_dir, "images", '100k', 'val')
            self.json_dir = os.path.join(data_dir, "labels", 'bdd100k_labels_images_val.json')
        if flag == 'test':
            self.img_dir = os.path.join(data_dir, "images", '100k', 'test')

        self.names = [name[:-4] for name in list(filter(lambda x: x.endswith(".jpg"), os.listdir(self.img_dir)))]
        self.label_data = json.load(open(self.json_dir, 'r', encoding='UTF-8'))
        self.label_list = label_list

    def __getitem__(self, index):
        """
        返回img和target
        :param idx:
        :return:
        """

        name = self.names[index]
        path_img = os.path.join(self.img_dir, name + ".jpg")
        # path_json = self.json_dir

        # load img
        img = Image.open(path_img).convert("RGB")

        # load boxes and label
        label_data = self.label_data
        points = label_data[index]['labels']
        boxes_list = list()
        labels_list = list()
        for point in points:
            if 'box2d' in point.keys():
                box = point['box2d']
                boxes_list.append([box['x1'], box['y1'], box['x2'], box['y2']])
                label = point['category']
                labels_list.append(self.label_list.index(label))
        boxes = torch.tensor(boxes_list, dtype=torch.float)
        labels = torch.tensor(labels_list, dtype=torch.long)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        if len(self.names) == 0:
            raise Exception("\ndata_dir:{} is a empty dir! Please checkout your path to images!".format(data_dir))
        return len(self.names)
