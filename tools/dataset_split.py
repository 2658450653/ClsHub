# -*- coding: utf-8 -*-
"""
将数据集划分为训练集，验证集，测试集
"""

import os
import random

import splitfolders

# 1.确定原图像数据集路径
dataset_dir = "/DATA/DATA/lzw/data/ScienceDataBank/"  ##原始数据集路径
# 2.确定数据集划分后保存的路径
split_dir = "/DATA/DATA/lzw/data/ScienceDataBank_split/"  ##划分后保存路径


# train:validation:test=8:1:1
splitfolders.ratio(input=dataset_dir,
                   output=split_dir,
                   seed=1337, ratio=(0.8, 0.1, 0.1))
