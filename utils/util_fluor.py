#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Macaca-Star
@File    ：util.py
@Author  ：Zauber
@Date    ：2025/3/1
"""
import os
import matplotlib.pyplot as plt
import ants
import cv2
import numpy as np
import yaml
from utils.util import reset_img

blockface_YAML_PATH = os.getcwd() + '/config/blockface_config.yaml'
blockface_CONFIG = yaml.safe_load(open(blockface_YAML_PATH, 'r'))
fluor_YAML_PATH = os.getcwd() + '/config/fluor_sections_config.yaml'
fluor_CONFIG = yaml.safe_load(open(fluor_YAML_PATH, 'r'))
MRI_YAML_PATH = os.getcwd() + '/config/MRI_config.yaml'
MRI_CONFIG = yaml.safe_load(open(MRI_YAML_PATH, 'r'))


def atlas_reg_ByT1w():
    # t1 = ants.image_read(MRI_CONFIG['MRI_file'])
    t1=ants.image_read('example/fluor_sections/MRI.nii.gz')
    tsfer = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/T1wlikeB_c.nii.gz')
    blockface=ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/b_recon_oc_scale_alignMRI.nii.gz')
    tmp_origin = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    atlas = ants.image_read('template/NMT/NMT_brain/D99_atlas_in_NMT_cortex.nii.gz')
    atlas1 = ants.image_read('template/NMT/NMT_brain/CHARM_1_in_NMT_v2.0_sym.nii.gz')
    atlas2 = ants.image_read('template/NMT/NMT_brain/SARM_2_in_NMT_v2.0_sym.nii.gz')
    atlas3 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_segmentation_edit.nii.gz')
    atlas4 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_cerebellum_mask.nii.gz')
    atlas5 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_segmentation.nii.gz')
    t1, tsfer, blockface, tmp, atlas, atlas1, atlas2,atlas3,atlas4,atlas5 = reset_img([t1, tsfer, blockface, tmp_origin, atlas, atlas1, atlas2,atlas3,atlas4,atlas5])
    tf1 = ants.registration(t1,tmp, 'SyN',
                            syn_metric='mattes',
                            reg_iterations=(40, 20, 0),flow_sigma=3,outprefix=fluor_CONFIG['output_dir']+'/reg3D/xfms/atlas_NMTtoT1w_')
    tmp_ = ants.apply_transforms(t1,tmp, tf1['fwdtransforms'],'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(tmp_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/NMT_inT1w.nii.gz')

    tf3 = ants.registration(t1,tsfer, 'SyN',
                            syn_metric='mattes',
                            reg_iterations=(40, 20, 0),flow_sigma=2,outprefix=fluor_CONFIG['output_dir']+'/reg3D/xfms/atlas_PItoT1w_')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(tf3['warpedmovout']))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/T1PI_inT1w.nii.gz')
    tsfer_ = ants.apply_transforms(t1,tsfer, tf3['fwdtransforms'], 'bSpline')

    tf2 = ants.registration(tsfer_,tmp_, 'SyN',
                            syn_metric='mattes',
                            reg_iterations=(40, 20,0),flow_sigma=3,outprefix=fluor_CONFIG['output_dir']+'/reg3D/xfms/atlas_NMTtoPIinT1w_')
    tmp_=tf2['warpedmovout']
    tmp_ = ants.apply_transforms(tsfer, tmp_, tf3['invtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(tmp_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/NMT_inblockface.nii.gz')

    ####################################################################
    atlas_ = ants.apply_transforms(t1, atlas, tf1['fwdtransforms'], 'multiLabel')
    atlas_.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/D99_inT1w.nii.gz')
    atlas_ = ants.apply_transforms(tsfer_, atlas_, tf2['fwdtransforms'], 'multiLabel')
    atlas_ = ants.apply_transforms(tsfer, atlas_, tf3['invtransforms'], 'multiLabel')
    atlas_.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/D99_inblockface.nii.gz')

    atlas1_ = ants.apply_transforms(t1, atlas1, tf1['fwdtransforms'], 'multiLabel')
    atlas1_.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/CHARM1_inT1w.nii.gz')
    atlas1_ = ants.apply_transforms(tsfer_, atlas1_, tf2['fwdtransforms'], 'multiLabel')
    atlas1_ = ants.apply_transforms(tsfer, atlas1_, tf3['invtransforms'], 'multiLabel')
    atlas1_.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/CHARM1_inblockface.nii.gz')

    atlas2_ = ants.apply_transforms(t1, atlas2, tf1['fwdtransforms'], 'multiLabel')
    atlas2_.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/SARM2_inT1w.nii.gz')
    atlas2_ = ants.apply_transforms(tsfer_, atlas2_, tf2['fwdtransforms'], 'multiLabel')
    atlas2_ = ants.apply_transforms(tsfer, atlas2_, tf3['invtransforms'], 'multiLabel')
    atlas2_.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/SARM2_inblockface.nii.gz')

    atlas3_ = ants.apply_transforms(t1, atlas3, tf1['fwdtransforms'], 'genericLabel')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(atlas3_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/segmentation_edit_inT1w.nii.gz')
    atlas3_ = ants.apply_transforms(tsfer_, atlas3_, tf2['fwdtransforms'], 'genericLabel')
    atlas3_ = ants.apply_transforms(tsfer, atlas3_, tf3['invtransforms'], 'genericLabel')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(atlas3_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/segmentation_edit_inblockface.nii.gz')

    atlas4_ = ants.apply_transforms(t1, atlas4, tf1['fwdtransforms'], 'genericLabel')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(atlas4_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/cerebellum_mask_inT1w.nii.gz')
    atlas4_ = ants.apply_transforms(tsfer_, atlas4_, tf2['fwdtransforms'], 'genericLabel')
    atlas4_ = ants.apply_transforms(tsfer, atlas4_, tf3['invtransforms'], 'genericLabel')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(atlas4_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/cerebellum_mask_inblockface.nii.gz')

    atlas5_ = ants.apply_transforms(t1, atlas5, tf1['fwdtransforms'], 'genericLabel')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(atlas5_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/segmentation_inT1w.nii.gz')
    atlas5_ = ants.apply_transforms(tsfer_, atlas5_, tf2['fwdtransforms'], 'genericLabel')
    atlas5_ = ants.apply_transforms(tsfer, atlas5_, tf3['invtransforms'], 'genericLabel')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(atlas5_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/segmentation_inblockface.nii.gz')

    img_ = ants.apply_transforms(tmp, t1, tf1['invtransforms'], 'bSpline')
    img_=ants.copy_image_info(tmp_origin,img_)
    img_.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/T1w_inNMT.nii.gz')

    img_ = ants.apply_transforms(t1, tsfer, tf3['fwdtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(img_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/T1PI_inT1w.nii.gz')
    img_ = ants.apply_transforms(tmp, img_, tf2['invtransforms'], 'bSpline')
    img_ = ants.apply_transforms(tmp, img_, tf1['invtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(img_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/T1PI_inNMT.nii.gz')

    blockface_ = ants.apply_transforms(t1, blockface, tf3['fwdtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(blockface_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/blockface_inT1w.nii.gz')
    blockface_ = ants.apply_transforms(tmp, blockface_, tf2['invtransforms'], 'bSpline')
    blockface_ = ants.apply_transforms(tmp, blockface_, tf1['invtransforms'], 'bSpline')
    img__ = ants.copy_image_info(tmp_origin, ants.image_clone(blockface_))
    img__.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/blockface_inNMT.nii.gz')


def atlas_reg_noT1w():
    tsfer = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/T1wlikeB_c.nii.gz')
    blockface=ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/b_recon_oc_scale_alignMRI.nii.gz')
    tmp_origin = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    atlas = ants.image_read('template/NMT/NMT_brain/D99_atlas_in_NMT_cortex.nii.gz')
    atlas1 = ants.image_read('template/NMT/NMT_brain/CHARM_1_in_NMT_v2.0_sym.nii.gz')
    atlas2 = ants.image_read('template/NMT/NMT_brain/SARM_2_in_NMT_v2.0_sym.nii.gz')
    atlas3 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_segmentation_edit.nii.gz')
    atlas4 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_cerebellum_mask.nii.gz')
    atlas5 = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_segmentation.nii.gz')
    tsfer, blockface, tmp, atlas, atlas1, atlas2,atlas3, atlas4, atlas5 = reset_img([tsfer, blockface, tmp_origin, atlas, atlas1, atlas2,atlas3, atlas4, atlas5])
    tf1 = ants.registration(tsfer,tmp, 'SyN',
                            syn_metric='mattes',
                            reg_iterations=(40, 40, 40),flow_sigma=3,outprefix=fluor_CONFIG['output_dir']+'/reg3D/xfms/atlas_PItoNMT_')
    img_ = ants.copy_image_info(tmp_origin, tf1['warpedmovout'])
    img_.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/NMT_inblockface.nii.gz')
    ###################################################################

    atlas_ = ants.apply_transforms(tsfer, atlas, tf1['fwdtransforms'], 'multiLabel')
    atlas_.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/D99_inblockface.nii.gz')

    atlas2_ = ants.apply_transforms(tsfer, atlas2, tf1['fwdtransforms'], 'multiLabel')
    atlas2_.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/SARM2_inblockface.nii.gz')

    atlas3_ = ants.apply_transforms(tsfer, atlas3, tf1['fwdtransforms'], 'multiLabel')
    atlas3_.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/segmentation_edit_inblockface.nii.gz')

    atlas4_ = ants.apply_transforms(tsfer, atlas4, tf1['fwdtransforms'], 'genericLabel')
    atlas4_.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/cerebellum_mask_inblockface.nii.gz')

    atlas5_ = ants.apply_transforms(tsfer, atlas5, tf1['fwdtransforms'], 'multiLabel')
    atlas5_.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/segmentation_inblockface.nii.gz')

    img_ = ants.apply_transforms(tmp, tsfer, tf1['invtransforms'], 'bSpline')
    img_=ants.copy_image_info(tmp_origin, img_)
    img_.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/T1PI_inNMT.nii.gz')

    blockface_ = ants.apply_transforms(tmp, blockface, tf1['invtransforms'], 'bSpline')
    blockface_ = ants.copy_image_info(tmp_origin, blockface_)
    blockface_.to_file(fluor_CONFIG['output_dir']+'/reg3D/atlas/blockface_inNMT.nii.gz')


def get_fslice_mask(img,hole_size=50):
    isPlot=False
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # ksize=5,5
    image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)
    img = cv2.cvtColor(shifted, cv2.COLOR_RGB2GRAY)
    # gray = util.invert(img)
    gray=img
    ret, mask = cv2.threshold(gray, 10, 255,  cv2.THRESH_BINARY)
    threshod_image_erode = cv2.erode(mask, kernel2, iterations=1)

    contours, _ = cv2.findContours(threshod_image_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= hole_size:
            cv_contours.append(contour)
        else:
            continue
    threshod_image_erode=cv2.fillPoly(threshod_image_erode, cv_contours, 255)
    plot_show(gray, threshod_image_erode, isPlot)
    ret, threshod_image = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)
    threshod_image=threshod_image_erode+threshod_image
    plot_show(gray, threshod_image, isPlot)
    threshod_image[threshod_image>0]=255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(threshod_image, cv2.MORPH_OPEN, kernel, iterations=2)
    plot_show(img, opening, isPlot)
    mask=opening
    mask[mask>0]=1
    return mask


def plot_show(image,image2,isPlot=False):
    if isPlot:
        fig = plt.figure(figsize=(12, 8), dpi=100)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(image,cmap='gray')
        ax1.set_title('image')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(image2,cmap='gray')
        ax2.set_title('segmentation')
        plt.show()

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range