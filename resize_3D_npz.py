# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/24 13:08
@Auth ： wongbooming
@File ：resize_3D_npz.py
@Explain :
"""

"""
@Time ： 2021/6/25 9:58
@Auth ： wangbooming
@File ：resize_3D.py
@IDE ：PyCharm
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



width = 200
height = 200
dim = 90
img_stack = np.load(r'E:\Yizhun-AI\data\SKI10\training\images\vol-021.npz')['vol']
im = Image.fromarray(img_stack[10])
print(np.shape(img_stack))
im.show()

img_stack_sm = np.zeros((dim, width, height))

for idx in range(dim):
    img = img_stack[idx, :, :]
    img_sm = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    img_stack_sm[idx, :, :] = img_sm

np.savez(r'E:\Yizhun-AI\data\SKI10\training\1',img_stack_sm)

print(np.shape(img_stack_sm))
im_resize = Image.fromarray(img_stack_sm[10])
im_resize.show()

new_im = np.load(r'E:\Yizhun-AI\data\SKI10\training\1.npz')
print(new_im.files)
print(np.shape(new_im))

