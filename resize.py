# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/24 12:58
@Auth ： wongbooming
@File ：resize.py
@Explain :
"""


from PIL import Image
import os
image_width = 512
image_height = 512


def fixed_size(filePath,savePath):
    """按照固定尺寸处理图片"""
    im = Image.open(filePath)
    out = im.resize((image_width, image_height), Image.NEAREST)
    # out.save(savePath)
    captcha = out.convert('RGB')
    captcha.save(savePath)


def changeSize():

    filePath = r'E:\seed\test\test_org_image'
    destPath = r'E:\seed\test\test_org_image_512'
    if not os.path.exists(destPath):
        os.makedirs(destPath)
    for root, dirs, files in os.walk(filePath):
        for file in files:
            if file[-1]=='g':
                fixed_size(os.path.join(filePath, file), os.path.join(destPath, file))
    print('Done')


if __name__ == '__main__':
    changeSize()