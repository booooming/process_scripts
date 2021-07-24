# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/24 12:56
@Auth ： wongbooming
@File ：deleate_empty_mask&image.py
@Explain : 删除空mask, 只有背景，没有其它类的mask以及对应的image
"""

# 删除没有只有白板的空mask以及对应的image

import os
import cv2


def deleate_mask(mask_paths, image_paths):
    for filename in os.listdir(mask_paths):
        mask_path = os.path.join(mask_paths, filename)
        img_path = os.path.join(image_paths, filename)
        img = cv2.imread(mask_path)
        dict = set()
        for i in range(img.shape[2]):
            for j in range(img.shape[0]):
                for k in range(img.shape[1]):
                    if img[j, k, i] not in dict:
                        dict.add(img[j, k, i])
        if dict != {0, 255}:
            print(dict)
            print(mask_path, end=",")
            os.remove(mask_path.replace("\\", "/"))
            os.remove(img_path.replace("\\", "/"))

                    # 读取的目录


deleate_mask(r"E:\seed\train\train_mask", r"E:\seed\train\train_org_image")
