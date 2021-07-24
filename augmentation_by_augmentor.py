# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/24 13:00
@Auth ： wongbooming
@File ：augmentation_by_augmentor.py
@Explain : 利用 augmentor 实现数据增强
"""

import Augmentor
import glob
import os
import random
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

train_path = r'E:\seed\train\test\image'
groud_truth_path = r'E:\seed\train\test\mask'
img_type = 'png'
train_tmp_path = r'E:\seed\train\test\tmp\image'
mask_tmp_path = r'E:\seed\train\test\tmp\image'


def start(train_path, groud_truth_path):
    train_img = glob.glob(train_path + '/*.' + img_type)
    masks = glob.glob(groud_truth_path + '/*.' + img_type)

    if len(train_img) != len(masks):
        print("trains can't match masks")
        return 0
    for file in os.listdir(train_path):
        file_name = file.split('.')[0]
        train_img_tmp_path = train_tmp_path + '/' + file_name
        if not os.path.lexists(train_img_tmp_path):
            os.mkdir(train_img_tmp_path)
        img = load_img(train_path + '/' + file_name + '.' + img_type)
        x_t = img_to_array(img)
        img_tmp = array_to_img(x_t)
        img_tmp.save(train_img_tmp_path + '/' + file_name + '.' + img_type)

        mask_img_tmp_path = mask_tmp_path + '/' + file_name
        if not os.path.lexists(mask_img_tmp_path):
            os.mkdir(mask_img_tmp_path)
        mask = load_img(groud_truth_path + '/' + file_name + '.' + img_type)
        x_l = img_to_array(mask)
        mask_tmp = array_to_img(x_l)
        mask_tmp.save(mask_img_tmp_path + '/' + file_name + '.' + img_type)
        print("%s folder has been created!" % file_name)


def doAugment():
    for file in os.listdir(train_path):
        file_name = file.split('.')[0]
        p = Augmentor.Pipeline(train_tmp_path + '/' + file_name)
        p.ground_truth(mask_tmp_path + '/' + file_name)
        p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)  # 旋转
        p.flip_left_right(probability=0.5)  # 按概率左右翻转
        p.zoom_random(probability=0.6, percentage_area=0.99)  # 随即将一定比例面积的图形放大至全图
        p.flip_top_bottom(probability=0.6)  # 按概率随即上下翻转
        p.random_distortion(probability=0.8, grid_width=10, grid_height=10, magnitude=20)  # 小块变形
        count = random.randint(40, 60)
        # print("\nNo.%s data is being augmented and %s data will be created" % (i, count))
        p.sample(count)
    print("Done")
    # print("%s pairs of data has been created totally" % sum)


start(train_path, groud_truth_path)
doAugment()