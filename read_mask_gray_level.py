# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/24 13:03
@Auth ： wongbooming
@File ：read_mask_gray_level.py
@Explain : 读取mask内有几种灰度值
"""


import cv2

img = cv2.imread(r"E:\seed\train\mask_512\0aGT6gN7.png")
print(img.shape)

dict = set()    # 创建空集合

for i in range(img.shape[2]):
    for j in range(img.shape[0]):
        for k in range(img.shape[1]):
            if img[j, k, i] not in dict:
                dict.add(img[j, k, i])

print(dict)