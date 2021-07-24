# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/24 13:06
@Auth ： wongbooming
@File ：read_image2txt_list.py
@Explain : 读取数据内图像名字到txt文本列表
"""

"""
@Time ： 2021/7/1 11:39
@Auth ： wangbooming
@File ：process_sys.py
@IDE ：PyCharm
"""
import numpy as np
import os
import SimpleITK as sitk
import random
from scipy import ndimage
from os.path import join
import config



class LITS_preprocess:
    def __init__(self, raw_dataset_path, fixed_dataset_path, args):
        self.raw_root_path = raw_dataset_path
        self.fixed_path = fixed_dataset_path
        self.classes = args.n_labels  # 分割类别数（只分割肝脏为2，或者分割肝脏和肿瘤为3）
        self.upper = args.upper
        self.lower = args.lower
        self.expand_slice = args.expand_slice  # 轴向外侧扩张的slice数量
        self.size = args.min_slices  # 取样的slice数量
        self.xy_down_scale = args.xy_down_scale
        self.slice_down_scale = args.slice_down_scale

        self.valid_rate = args.valid_rate


    def write_train_val_name_list(self):
        # images_list = os.listdir(join(self.raw_root_path, "image"))
        # data_num = len(images_list)
        # print('the fixed dataset total numbers of samples is :', data_num)
        # # random.shuffle(images_list)
        #
        # self.images_write_name_list(images_list, "image_list.txt")

        labels_list = os.listdir(join(self.raw_root_path, "mask"))
        data_num = len(labels_list)
        random.shuffle(labels_list)
        labels_list_train = labels_list[:int(data_num * 0.6)]
        print(len(labels_list_train))
        labels_list_test = labels_list[int(data_num * 0.6):int(data_num * 0.8)]
        print(len(labels_list_test))
        labels_list_val = labels_list[int(data_num * 0.8):]
        print(len(labels_list_val))

        print('the fixed dataset total numbers of samples is :', data_num)
        # random.shuffle(labels_list)
        self.labels_write_name_list(labels_list, "mask_list_all.txt")
        self.labels_write_name_list(labels_list_val, "mask_list_val.txt")
        self.labels_write_name_list(labels_list_train, "mask_list_train.txt")
        self.labels_write_name_list(labels_list_test, "mask_list_test.txt")

        self.images_write_name_list(labels_list, "image_list_all.txt")
        self.images_write_name_list(labels_list_val, "image_list_val.txt")
        self.images_write_name_list(labels_list_train, "image_list_train.txt")
        self.images_write_name_list(labels_list_test, "image_list_test.txt")

    def images_write_name_list(self, name_list, file_name):
        f = open(join(self.fixed_path, file_name), 'w')
        for name in name_list:
            ct_path = os.path.join(self.raw_root_path, 'image', name)
            f.write(ct_path + "\n")
        f.close()

    def labels_write_name_list(self, name_list, file_name):
        f = open(join(self.fixed_path, file_name), 'w')
        for name in name_list:
            seg_path = os.path.join(self.raw_root_path, 'mask', name)
            f.write(seg_path + "\n")
        f.close()


if __name__ == '__main__':
    raw_dataset_path = r'E:\Yizhun-AI\data\ACL_aclseg'
    fixed_dataset_path = r"E:\Yizhun-AI\data\ACL_aclseg_txt"

    args = config.args
    tool = LITS_preprocess(raw_dataset_path, fixed_dataset_path, args)
    tool.write_train_val_name_list()  # 创建索引txt文件
