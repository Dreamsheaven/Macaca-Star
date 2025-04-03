#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Macaca-Star
@File    ：PI_MRI.py
@Author  ：Zauber
@Date    ：2024/6/3
"""
from utils.fMOST_PI_preproc import tif_to_nii, normalize_to_8bit, denoise_img, remove_artifact, mas_cerebellum


def PI_preproc():
    # tif_to_nii()
    # normalize_to_8bit()
    # denoise_img()
    mas_cerebellum()
    remove_artifact()