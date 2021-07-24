# -*- coding: utf-8 -*-
"""
@Time ： 2021/7/24 13:11
@Auth ： wongbooming
@File ：Interpolation_3D_handle.py
@Explain : 手工3D插值，很慢
"""

import math

import numpy as np

def Interpolation3D(src_img, dst_size):
    srcZ, srcY, srcX = src_img.shape
    dst_img = np.zeros(shape=dst_size, dtype=np.int16)
    new_Z, new_Y, new_X = dst_img.shape
    # print("插值后图像的大小", dst_img.shape)
    factor_z = srcZ / new_Z
    factor_y = srcY / new_Y
    factor_x = srcX / new_X

    for z in range(new_Z):
        for y in range(new_Y):
            for x in range(new_X):

                src_z = z * factor_z
                src_y = y * factor_y
                src_x = x * factor_x

                src_z_int = math.floor(z * factor_z)
                src_y_int = math.floor(y * factor_y)
                src_x_int = math.floor(x * factor_x)

                w = src_z - src_z_int
                u = src_y - src_y_int
                v = src_x - src_x_int
                # 判断是否查出边界
                if src_x_int + 1 == srcX or src_y_int + 1 == srcY or src_z_int + 1 == srcZ:
                    dst_img[z, y, x] = src_img[src_z_int, src_y_int, src_x_int]
                else:
                    C000 = src_img[src_z_int, src_y_int, src_x_int]
                    C001 = src_img[src_z_int, src_y_int, src_x_int + 1]
                    C011 = src_img[src_z_int, src_y_int + 1, src_x_int + 1]
                    C010 = src_img[src_z_int, src_y_int + 1, src_x_int]
                    C100 = src_img[src_z_int + 1, src_y_int, src_x_int]
                    C101 = src_img[src_z_int + 1, src_y_int, src_x_int + 1]
                    C111 = src_img[src_z_int + 1, src_y_int + 1, src_x_int + 1]
                    C110 = src_img[src_z_int + 1, src_y_int + 1, src_x_int]
                    dst_img[z, y, x] = C000 * (1 - v) * (1 - u) * (1 - w) + \
                                       C100 * v * (1 - u) * (1 - w) + \
                                       C010 * (1 - v) * u * (1 - w) + \
                                       C001 * (1 - v) * (1 - u) * w + \
                                       C101 * v * (1 - u) * w + \
                                       C011 * (1 - v) * u * w + \
                                       C110 * v * u * (1 - w) + \
                                       C111 * v * u * w

    return dst_img
