# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/24 13:01
@Auth ： wongbooming
@File ：generate_mask.py
@Explain : 根据分割模型权重生成分割mask
"""


import torch
import time

import torchvision
from PIL.Image import NEAREST
from albumentations.pytorch.functional import img_to_tensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import unet, nested_unet, loss_function, xnet, nfn_plus
from datasets import binary_glomerulus, brain_mri, chest_xray, skin_lesion, chaos, glomerulus, noisy_brain_mri, \
    noisy_chaos
from torchvision import transforms
import os
from PIL import Image
import argparse
import numpy as np
from utils.metrics import compute_metrics
from utils.tools import create_directory
from collections import OrderedDict
import cv2

# 选择网络模型

net = unet.UNet(num_classes=2, in_channels=3)
# net = unet.UNet(num_classes=2, in_channels=3, is_attention=True)

# 加载模型
ckpt = torch.load(r"E:\seed\traingingrecords\seg_newdata\unet_seed_data_512_Dice_0723_1754_epoch_299.pkl")
ckpt = ckpt['model_state_dict']
new_state_dict = OrderedDict()
for k, v in ckpt.items():
    name = k[7:]  # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
    new_state_dict[name] = v  # 新字典的key值对应的value为一一对应的值。

net.load_state_dict(new_state_dict)
net.eval()

pre_data_path = r"E:\seed\test\test_org_image"
dst_path = r"E:\seed\submit\submit_unet_dice_512\result\result"
image_list = []
for file in os.listdir(pre_data_path):
    image_list.append(os.path.join(pre_data_path, file))

with torch.no_grad():
    i = 0
    for image in image_list:
        print(i)
        i += 1
        name = image.split("\\")[-1]
        # print(name)
        image = Image.open(image).convert('RGB')
        ori_size = image.size
        image = image.resize((512, 512), resample=NEAREST)
        # image = img_to_tensor(image)
        image = transforms.ToTensor()(image)
        # print(image.shape)  # [3, 224, 224]
        image = Variable(torch.unsqueeze(image, dim=0).float(), requires_grad=False)
        # print(image.shape)  # [1, 3, 224, 224]
        outputs = net(image)    # # [1, 2, 224, 224], 此2应该为类别数
        # softmax = torch.nn.Softmax(dim=1)
        # outputs = softmax(outputs)
        outputs = outputs.squeeze(0)    # [2, 224, 224]

        # mask = np.argmax(outputs, axis=0)
        mask = torch.max(outputs, 0)[1].cpu().numpy()
        # print(mask.max())
        # print(mask.shape)
        a = np.zeros(mask.shape, np.float32)
        a[mask == 1] = 255
        # a[mask == 0] = 0
        # for i in range(np.shape(a)[0]):
        im = Image.fromarray(np.uint8(a[:, :]))
        im = im.resize(ori_size, resample=NEAREST)     # 需要使用最近邻插值
        im.convert('RGB').save(dst_path + "\\" + str(name))


