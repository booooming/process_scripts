# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/24 13:07
@Auth ： wongbooming
@File ：npz.py
@Explain :
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import cv2
from PIL import Image
import torch
import torch.nn.functional as F

# src_path = r"E:\Yizhun-AI\data\SKI10\training\labels"
# dst_path = r"E:\Yizhun-AI\data\test"
# # dict = {str(0.235),}

# dim1_list = []
# dim2_list = []
# dim3_list = []
# for root, dirs, files in os.walk(src_path):
#     for file in files:
#         # print("files",files)
#         # print("file",file)
#         a = np.load(os.path.join(root,file))
#         arr = a['vol']
#         # print(np.shape(arr))
#         # print(np.shape(arr)[2])
#         n = len(arr)
#         dim1_list.append(n)
#         p = np.shape(arr)[1]
#         dim2_list.append(p)
#         q = np.shape(arr)[2]
#         dim3_list.append(q)
#         # for l in range(n):
#         #     p = np.shape(arr)[1]
#         #     q = np.shape(arr)[2]
#         #     # print(p)
#         #     # print(q)
#         #     for i in range(p):
#         #         for j in range(q):
#         #             # print(str(arr[l][i][j]))
#         #             if str(arr[l][i][j]) not in dict:
#         #                 dict.add(str(arr[l][i][j]))
#         # print(dict)
# print(dim1_list)
# print(min(dim1_list))
# print(max(dim1_list))
# print(dim2_list)
# print(min(dim2_list))
# print(max(dim2_list))
# print(dim3_list)
# print(min(dim3_list))
# print(max(dim3_list))

# for i in range(n):
#
#     # for x in arr[i]:
#     #     if x not in dict:
#     #         dict.add("x")
#     im = Image.fromarray(arr[i])
#     print(str(file[:-4]))
#     im.save(dst_path + "\\" + str(file[:-4]) + "-" + str(i) + ".png")
# print(dict)

# a = np.load(r'data\promise12\npy_image\X_train.npy')
# print(np.shape(a))
# for i in range(600):
#     # print(a[i])
#     im = Image.fromarray(a[i])
# im.show()

# a = np.load(r'E:\Yizhun-AI\data\SKI10\training\images\vol.npy')
# print(np.shape(a))
# for i in range(30):
#     # print(a[i])
#     im = Image.fromarray(a[i])
# im.show()

# 查看images与labels维度是否一致，挑选出不一致的
# for i in range(10,60):
#     a = np.load(r'E:\Yizhun-AI\data\SKI10\training\labels\vol-0' + str(i) + '.npz')['vol']
#     # print("a",a)
#     # print(np.shape(a[0]))
#     # if np.shape(a)!=np.shape(b):
#     print(i)
#     print(np.shape(a))
#     print("*****")


# a = np.load(r'E:\Imageseg_model\ShinThighBone_seg\SKI10\training\labels\vol-001.npz')['vol']
# # print("a",a)
# print(np.shape(a))
# order = 22
# print(np.shape(a[order]))
# im = Image.fromarray(a[order])
# im.show()
# n = len(a)
# jin191 = 0
# gu63 = 0
# dict = {'0','127', '255'}
# for l in range(n):
#     p = np.shape(a)[1]
#     q = np.shape(a)[2]
#
#     for i in range(p):
#         for j in range(q):
#             # print(str(arr[l][i][j]))
#             if str(a[l][i][j]) == "191":
#                 a[l][i][j] = "1"
#                 jin191 += 1
#                 # print("1")
#             elif str(a[l][i][j]) == "63":
#                 a[l][i][j] = "2"
#                 gu63 += 1
#                 # print("2")
#             elif str(a[l][i][j]) in dict:
#                 a[l][i][j] = "0"
#                 # print("0")
#
# im_revise = Image.fromarray(a[order])
# im_revise.show()
# with open(r'E:\Yizhun-AI\data\numpy' + str(order) + "_revise" + '.txt', 'a') as f4:
#     np.set_printoptions(precision=3)
#     np.savetxt(f4, im_revise, delimiter='\t', newline='\n',fmt='%.02f')
# f4.close()
# print(jin191)
# print(gu63)
# dst_path = r"E:\Yizhun-AI\data\revise\\"
# for i in range(n):
#     im = Image.fromarray(a[i])
#     im.save(dst_path + str(i) + ".png")


# np.set_printoptions(suppress=True)
# np.set_printoptions(precision=4)
# np.set_printoptions(threshold=np.inf)
# print("a:",a[i])


# 读取展示result文件图像
# b = np.load(r'ACL/testing/labels/1.3.12.2.1107.5.2.32.35183.2013052207595550789910230.0.0.0.npz')
# # print(b.files)
# # print(np.shape(b['arr_0']))
# res_path = r"E:\Yizhun-AI\data\compare\result-1.3.12.2.1107.5.2.32.35183.2013052207595550789910230.0.0.0\real"
# b = b["arr_0"]
# # b = b["mask"]
# print(np.shape(b))
# print(b[0])
#
#
# def process_seg(a):  # 替换标签，否则显示一团黑
#     b = np.zeros(a.shape, np.float32)
#     b[a == 1] = 125
#     b[a == 2] = 255
#     return b
#
#
# for i in range(np.shape(b)[0]):
#     b[i] = process_seg(b[i])
#     im = Image.fromarray(b[i])
#     im.save(res_path + "\\" + str(i) + ".png")

# 读取ACL_image数据展示正常标签图像
# b = np.load(r'E:\Yizhun-AI\data\ACL_aclseg\image\1.3.12.2.1107.5.2.32.35183.2013122719075371726513067.0.0.0.npz')['array']
# # print(b.files)
# res_path = r"E:\Yizhun-AI\data\ACL_test\1.3.12.2.1107.5.2.32.35183.2013122719075371726513067.0.0.0\image"
# print(np.shape(b))
# print(np.shape(b)[2])
# for i in range(np.shape(b)[2]):
#     im = Image.fromarray(b[:,:,i])
#     im.convert('RGB').save(res_path + "\\" + str(i) + ".png")

# 读取ACL_mask数据展示正常标签图像
b = np.load(r'E:\Yizhun-AI\data\visual\visual_seg.npz')['arr_0']
# print(b.files)
res_path = r"E:\Yizhun-AI\data\visual\visual_seg"

res = {"0.1"}
for i in range(np.shape(b)[0]):
    # im = Image.fromarray(b[i, :, :])
    # im.convert('RGB').save(res_path + "\\" + str(i) + "_mask_tri" + ".png")
    for m in range(np.shape(b)[1]):
        for n in range(np.shape(b)[2]):
            if str(b[i][m][n]) not in res:
                # print("ii")
                res.add(str(b[i][m][n]))

print(res)

a = np.zeros(b.shape, np.float32)
a[b == 1] = 125
a[b == 2] = 255
for i in range(np.shape(a)[0]):
    im = Image.fromarray(np.uint8(a[i, :, :]))
    im.convert('RGB').save(res_path + "\\" + str(i) + "_mask" + ".png")
# a = b.T
# # print("转置后seg_array:", seg_array.shape)
# seg_array = torch.FloatTensor(a).unsqueeze(0).unsqueeze(0)
# seg_array_resize = F.interpolate(seg_array, size=([32, 128, 128]), mode="nearest")
# seg_array_resize = seg_array_resize.squeeze(0).squeeze(0)
# seg_array_resize = seg_array_resize.numpy()
#
# dict = {'2.125'}
#
# # a = np.zeros(b.shape, np.float32)
# # a[b == 1] = 125
# # a[b == 2] = 255
#
# print(seg_array_resize.shape)
# for i in range(np.shape(seg_array_resize)[0]):
#     # im = Image.fromarray(seg_array_resize[i, :, :])
#     # im.convert('RGB').save(res_path + "\\" + str(i) + "_mask_tri" + ".png")
#     for m in range(np.shape(seg_array_resize)[1]):
#         for n in range(np.shape(seg_array_resize)[2]):
#             if str(seg_array_resize[i][m][n]) not in dict:
#                 dict.add(str(seg_array_resize[i][m][n]))
#
# print(dict)
# print(len(dict))
print("**********")

seg_array_resize = np.load(r'E:\Yizhun-AI\data\visual\visual_at.npz')['arr_0']
# seg_array_resize = seg_array_resize.unsqueeze(0)
# print(seg_array_resize.files)
print(seg_array_resize.shape)
res_path = r"E:\Yizhun-AI\data\visual\visual_seg"

for i in range(np.shape(seg_array_resize)[0]):
    im = Image.fromarray(seg_array_resize[i, :, :])
    im.convert('RGB').save(res_path + "\\" + str(i) + "_image" + ".png")

# seg_array_resize = seg_array_resize.T
#
# seg_array = torch.FloatTensor(seg_array_resize).unsqueeze(0).unsqueeze(0)
# seg_array_resize = F.interpolate(seg_array, size=([32, 128, 128]), mode="trilinear", align_corners=True)
# seg_array_resize = seg_array_resize.squeeze(0).squeeze(0)
# seg_array_resize = seg_array_resize.numpy()
#
# for i in range(np.shape(seg_array_resize)[0]):
#     im = Image.fromarray(seg_array_resize[i, :, :])
#     im.convert('RGB').save(res_path + "\\" + str(i) + "_image_tri" + ".png")

# 查看标签像素值

# a = np.load(r'experiments/UNet0.001/result_5/result-1.3.12.2.1107.5.2.32.35183.2013052207595550789910230.0.0.0.npz')['arr_0']
# n = np.shape(a)[2]
# dict = {'2.125'}
# for l in range(n):
#     p = np.shape(a)[0]
#     q = np.shape(a)[1]
#     for i in range(p):
#         for j in range(q):
#             if str(a[i][j][l]) not in dict:
#                 print("ii")
#                 dict.add(str(a[i][j][l]))
# print(dict)

# 查看二维图像灰度像素值

# a = cv2.imread(r'E:\Yizhun-AI\data\ACL_test\1.3.12.2.1107.5.2.32.35183.2013122719075371726513067.0.0.0\mask\6.png', 0)
# print(a.shape)
# dict = {'2.125'}
# for i in range(320):
#     for j in range(320):
#         if str(a[i][j]) not in dict:
#             print("ii")
#             dict.add(str(a[i][j]))
# print(dict)

# print(np.shape(a))
# b = a[60]
# plt.imshow(b)
# plt.show()
# im = Image.fromarray(b)
# im.show()
# print('shape b = ', np.shape(b))


# np.shape(b)
# c = np.tile(b, [1,3])
# c.shape
# plt.imshow(c)
# plt.show()
# im = Image.fromarray(c)
# im.show()
#
# np.max(b)
#
# np.max(b)
# d = b/np.max(b)*255
# im = Image.fromarray(d)
# im.show()
# f = np.array(d, dtype = np.uint8)
# im = Image.fromarray(f)
# im.show()
# type(f)
# c = np.tile(b, [1,3])
# c.shape
# plt.imshow(c)
# plt.show()
# im = Image.fromarray(c)
# im.show()

# import numpy as np
# cat_data = np.load(r'E:\Yizhun-AI\data\SKI10\training\images\vol-038.npz')
# print(cat_data.files)
# data = cat_data['vol']
# print(data)
# import numpy as np
# cat_data = np.load(r'E:\Yizhun-AI\data\SKI10\training\labels\vol-001.npz'['vol'])
# print(cat_data)
