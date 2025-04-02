#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Macaca-Star
@File    ：fMOST_PI_preproc.py
@Author  ：Zauber
@Date    ：2024/6/3
"""
import os
import numpy as np
import utils.Logger as loggerz
import tifffile
import ants
import yaml

from utils.util import horizontal, sagittal

YAML_PATH = os.getcwd() + '/config/fMOST_PI_config.yaml'
fMOST_PI_CONFIG = yaml.safe_load(open(YAML_PATH, 'r'))

def tif_to_nii():
    logger = loggerz.get_logger()
    logger.info('tif to nii.gz')
    tif_file = tifffile.imread(fMOST_PI_CONFIG['subject_dir'])
    nmt = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    tmp = tif_file
    tif = ants.from_numpy(tmp)
    tif.set_spacing(fMOST_PI_CONFIG['spacing'])
    tif.set_origin(nmt.origin)
    tif.set_direction(nmt.direction)
    tif.to_file(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI.nii.gz')


def normalize_to_8bit():
    logger = loggerz.get_logger()
    logger.info('normalize to 8bit')
    img = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI.nii.gz')
    img_data = img.numpy()
    # Exclude abnormal intensities
    percent_upper = np.percentile(img_data, 95)
    img_ = img_data[img_data < percent_upper]
    if np.max(img_) > 255.0:
        space = np.max(img_) - np.min(img_)
        norm = (img - np.min(img.numpy()[:, :, :].all())) * 255 / (space + 1E-6)
    else:
        # If the maximum value is already less than 255, there is no need to normalize to 8-bit
        print('max < 255')
        norm = img
    ants.image_write(norm, fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit.nii.gz')

def mas_cerebellum():
    logger = loggerz.get_logger()
    logger.info('Remove cerebellum')
    fix = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit.nii.gz')
    fix_ = ants.resample_image(fix, (0.25, 0.25, 0.25), interp_type=4)
    move = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/NMT_v2.0_asym_brain_L.nii.gz')
    atlas = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/cerebellum_L.nii.gz')
    pi_affine_transform = ants.registration(fix_, move, type_of_transform='SyN', reg_iterations=(40, 20, 0))
    atlas_ = ants.apply_transforms(fix_, atlas, pi_affine_transform['fwdtransforms'], 'multiLabel')
    # atlas_ = ants.morphology(atlas_affine, operation='dilate', radius=3, mtype='binary', shape='ball')
    ants.image_write(atlas_,fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/atlas/cerebellum_mask_in_PI_0.25mm.nii.gz')
    pi_affine_transform = ants.registration(fix, fix_, type_of_transform='Affine', reg_iterations=(40, 20, 0))
    atlas_h = ants.apply_transforms(fix, atlas_, pi_affine_transform['fwdtransforms'], 'multiLabel')
    mas = ants.mask_image(fix, atlas_h, 0)
    ants.image_write(mas, fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/tmp/PI_rmc.nii.gz')
    ants.image_write(atlas_h, fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/tmp/cerebellum_mask.nii.gz')


def denoise_img():
    logger = loggerz.get_logger()
    logger.info('denoise the fMOST PI')
    pi = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit.nii.gz')
    # Denoise the fMOST PI (using a simple Gaussian filter)
    pi_denoise = ants.smooth_image(pi, sigma=1, max_kernel_width=5)
    pi_denoise.to_filename(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit_dn.nii.gz')


def remove_artifact():
    logger = loggerz.get_logger()
    logger.info('remove striping artifact')
    # horizontal()
    sagittal()