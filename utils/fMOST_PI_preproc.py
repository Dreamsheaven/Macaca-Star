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
import albumentations as A
from utils.util import horizontal, sagittal, crop_brain, log, atlas_reg_ByT1w, atlas_reg_noT1w

YAML_PATH = os.getcwd() + '/config/fMOST_PI_config.yaml'
fMOST_PI_CONFIG = yaml.safe_load(open(YAML_PATH, 'r'))
MRI_YAML_PATH = os.getcwd() + '/config/MRI_config.yaml'
MRI_CONFIG = yaml.safe_load(open(MRI_YAML_PATH, 'r'))

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
    fix_.to_file(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/tmp/PI_0.25mm.nii.gz')
    move = ants.image_read(os.getcwd() + '/template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    atlas = ants.image_read(os.getcwd() + '/template/NMT/NMT_brain/NMT_v2.0_sym_cerebellum_mask.nii.gz')
    move=crop_brain(move)
    atlas = crop_brain(atlas)
    pi_affine_transform = ants.registration(fix_, move, type_of_transform='SyN', reg_iterations=(40, 20, 0))
    atlas_ = ants.apply_transforms(fix_, atlas, pi_affine_transform['fwdtransforms'], 'multiLabel')
    atlas_ = ants.morphology(atlas_, operation='dilate', radius=3, mtype='binary', shape='ball')
    ants.image_write(atlas_,fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/atlas/cerebellum_mask_in_PI_0.25mm.nii.gz')
    pi_affine_transform = ants.registration(fix, fix_, type_of_transform='Affine', reg_iterations=(40, 20, 0))
    atlas_h = ants.apply_transforms(fix, atlas_, pi_affine_transform['fwdtransforms'], 'multiLabel')
    mas = ants.mask_image(fix, atlas_h, 0)
    ants.image_write(mas, fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/tmp/PI_rmc.nii.gz')
    ants.image_write(atlas_h, fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/atlas/cerebellum_mask.nii.gz')


def denoise_img():
    logger = loggerz.get_logger()
    logger.info('denoise the fMOST PI')
    pi = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit_rm.nii.gz')
    # Denoise the fMOST PI (using a simple Gaussian filter)
    pi_denoise = ants.smooth_image(pi, sigma=1, max_kernel_width=2)
    pi_denoise.to_filename(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit_rm_dn.nii.gz')


def remove_artifact():
    logger = loggerz.get_logger()
    logger.info('remove striping artifact')
    horizontal()
    sagittal()

def intensity_c():
    logger = loggerz.get_logger()
    logger.info('Intensity correction')
    img = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit_rm_dn.nii.gz')
    img=ants.iMath_normalize(img)*255
    img_data=img.numpy()
    if fMOST_PI_CONFIG['intensity_correction']:
        n = 50
        tmp = img.numpy() + 2.0
        matrix = np.where(tmp < n)
        for i in range(0, img.shape[2]):
            tmp_ = tmp[:, :, i]
            tmp[:, :, i] = log(n, tmp_) * n
        tmp[matrix] = img_data[matrix]
        tmp_morm = ants.from_numpy(tmp)
        tmp_morm.set_spacing(img.spacing)
        tmp_morm.set_direction(img.direction)
        tmp_morm.set_origin(img.origin)
        ants.image_write(tmp_morm, fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit_rm_dn_ic.nii.gz')
    else:
        ants.image_write(img, fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit_rm_dn_ic.nii.gz')

def clahe_image():
    logger = loggerz.get_logger()
    logger.info('Image Enhancement')
    pi = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit_rm_dn_ic.nii.gz')
    if fMOST_PI_CONFIG['clahe']:
        logger.info('START PI clahe')
        CLAHE = A.CLAHE(tile_grid_size=(8, 8), always_apply=False, p=0.5)
        spacing = pi.spacing
        direction = pi.direction
        pi = pi.numpy()
        space = np.max(pi) - np.min(pi)
        norm = (pi - np.min(pi)) * 255 / space
        pi = norm.astype(np.uint8)
        for i in range(0, pi.shape[0]):
            pi_splice = pi[i, :, :]
            pi_splice = CLAHE.apply(pi_splice,clip_limit= 2.0)
            pi[i, :, :] = pi_splice
        pi = pi.astype(np.uint8)
        pi_ = ants.from_numpy(pi)
        pi_.set_spacing(spacing)
        pi_.set_direction(direction)
    else:
        pi_ = pi
    ants.image_write(pi_, fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit_rm_dn_ic_.nii.gz')


def PI_alignNMT():
    logger = loggerz.get_logger()
    logger.info('fMOST PI align to NMT')
    img=ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/PI_8bit_rm_dn_ic_.nii.gz')
    img_=ants.resample_image(img,(0.25,0.25,0.25),use_voxels=False,interp_type=4)
    img_.to_file(fMOST_PI_CONFIG['output_dir'] + '/fMOST_PI/tmp/PI_0.1mm.nii.gz')
    nmt=ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    nmt=crop_brain(nmt)
    img_=ants.from_numpy(img_.numpy())
    nmt_=ants.from_numpy(nmt.numpy())

    t=ants.registration(nmt_,img_,'Similarity',aff_metric='GC',outprefix=fMOST_PI_CONFIG['output_dir']+'/reg/xfms/PItoNMT_')
    img_=ants.apply_transforms(nmt_,img_,t['fwdtransforms'],'bSpline')
    # img_ = ants.iMath_truncate_intensity(img_, 0, 0.98)
    img_ = ants.n4_bias_field_correction(img_, shrink_factor=8)
    img_ = ants.n4_bias_field_correction(img_, shrink_factor=4)
    img_ = ants.n4_bias_field_correction(img_, shrink_factor=2)
    img_.set_direction(nmt.direction)
    img_.set_spacing(nmt.spacing)
    img_.set_origin(nmt.origin)
    img_.to_file(fMOST_PI_CONFIG['output_dir']+'/reg/PI_alignNMT.nii.gz')


def correct_T1like():
    logger = loggerz.get_logger()
    logger.info('correct T1like')
    img=ants.image_read(fMOST_PI_CONFIG['output_dir']+'/reg/PI_alignNMT.nii.gz')
    t1like = ants.image_read(fMOST_PI_CONFIG['output_dir'] + '/reg/T1likePI.nii.gz')
    mask=ants.get_mask(img,2)
    t1like=ants.mask_image(t1like,mask)
    pi_inv=ants.new_image_like(t1like,t1like.numpy())
    pi_data=img.numpy()[:,:,:]
    pi_data_inv=(np.max(pi_data)-pi_data)* mask[:, :,:].numpy()
    pi_data_inv[pi_data_inv<0]=0
    pi_inv[:, :, :] = pi_data_inv
    t = ants.registration(pi_inv, t1like, 'SyN',syn_metric='mattes',reg_iterations=(40,40,40,40),syn_sampling=32)
    t1like_=ants.apply_transforms(img,t1like,t['fwdtransforms'],'bSpline')
    t1like_.to_file(fMOST_PI_CONFIG['output_dir']+'/reg/T1likePI_c.nii.gz')


def fMOST_PI_3Dreg():
    logger = loggerz.get_logger()
    logger.info('fMOST PI register to NMT')
    if MRI_CONFIG['MRI-guided']:
        logger.WARNING('MRI-guided registration')
        atlas_reg_ByT1w()
    else:
        logger.WARNING('no MRI-guided registration')
        atlas_reg_noT1w()

