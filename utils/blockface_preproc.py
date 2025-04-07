import os
import shutil
import albumentations as A
import ants
import numpy as np
import utils.Logger as loggerz
from utils.util import touint8
import yaml
from utils.util_fluor import atlas_reg_ByT1w, atlas_reg_noT1w, get_maskBywatershed, centerxy_img

blockface_YAML_PATH = os.getcwd() + '/config/blockface_config.yaml'
blockface_CONFIG = yaml.safe_load(open(blockface_YAML_PATH, 'r'))
fluor_YAML_PATH = os.getcwd() + '/config/fluor_sections_config.yaml'
fluor_CONFIG = yaml.safe_load(open(fluor_YAML_PATH, 'r'))
MRI_YAML_PATH = os.getcwd() + '/config/MRI_config.yaml'
MRI_CONFIG = yaml.safe_load(open(MRI_YAML_PATH, 'r'))


def intensity_c():
    logger = loggerz.get_logger()
    logger.info('Intensity correction')
    CLAHE = A.CLAHE(clip_limit=(1.0, 2.0), tile_grid_size=(10, 10), always_apply=False, p=0.5)
    img=ants.image_read(blockface_CONFIG['subject_dir'])
    for i in range(img.shape[1]):
        fslice=img[:,i,:].numpy()
        fslice=touint8(fslice)
        fslice=CLAHE.apply(fslice,clip_limit=2.0)
        img[:, i, :]=fslice
    img_=ants.denoise_image(img,ants.get_mask(img))
    img_.to_file(fluor_CONFIG['output_dir']+'/blockface/b_recon_oc_scale_clahe.nii.gz')

def b_alignMRI():
    logger = loggerz.get_logger()
    logger.info('blockface align to MRI')
    b = ants.image_read(fluor_CONFIG['output_dir']+'/blockface/b_recon_oc_scale_clahe.nii.gz')
    b=ants.from_numpy(b.numpy())
    nmt = ants.image_read('template/NMT/NMT_brain/NMT_v2.0_sym_SS.nii.gz')
    nmt_=ants.from_numpy(nmt.numpy())
    t2 = ants.registration(nmt_, b, 'Affine', aff_metric='GC',aff_sampling=100)
    b_ = ants.apply_transforms(nmt_, b, t2['fwdtransforms'],'bSpline')
    shutil.copyfile(t2['fwdtransforms'][0], fluor_CONFIG['output_dir']+'/reg3D/xfms/b_regt1.mat')
    mask=ants.get_mask(b_,10)
    b_=ants.mask_image(b_,mask)
    b_=ants.denoise_image(b_,mask)
    b_=ants.n4_bias_field_correction(b_,mask,shrink_factor=4)
    b_ = ants.n4_bias_field_correction(b_, mask, shrink_factor=2)
    b_=ants.copy_image_info(nmt,b_)
    b_.to_file(fluor_CONFIG['output_dir'] + '/reg3D/b_recon_oc_scale_alignMRI.nii.gz')

def correct_t1like():
    logger = loggerz.get_logger()
    logger.info('blockface align to MRI')
    tsfer=ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/T1likeBlockface.nii.gz')
    b=ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/b_recon_oc_scale_alignMRI.nii.gz')
    mask=ants.get_mask(b,10)
    img = ants.mask_image(tsfer, mask)
    img.to_file(fluor_CONFIG['output_dir']+'/reg3D/T1wlikeB_c.nii.gz')

def blockface_3Dreg():
    logger = loggerz.get_logger()
    logger.info('fMOST PI register to NMT')
    if MRI_CONFIG['MRI-guided']:
        logger.warning('MRI-guided registration')
        atlas_reg_ByT1w()
    else:
        logger.warning('no MRI-guided registration')
        atlas_reg_noT1w()


def b_invetalignMRI():
    logger = loggerz.get_logger()
    logger.info('blockface inverse align to MRI')
    isAffine=False
    b_origin = ants.image_read(fluor_CONFIG['output_dir'] + '/blockface/b_recon_oc_scale_clahe.nii.gz')
    b=ants.from_numpy(b_origin.numpy())
    b_alignMRI = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/b_recon_oc_scale_alignMRI.nii.gz')
    b_ = ants.from_numpy(b_alignMRI.numpy())
    atlas=ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/atlas/D99_inblockface.nii.gz')
    atlas1 = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/atlas/segmentation_inblockface.nii.gz')
    atlas2 = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/atlas/SARM2_inblockface.nii.gz')
    atlas4 = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/atlas/cerebellum_mask_inblockface.nii.gz')
    atlas6 = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/atlas/CHARM1_inblockface.nii.gz')
    atlas7 = ants.image_read(fluor_CONFIG['output_dir'] + '/reg3D/atlas/segmentation_edit_inblockface.nii.gz')
    nmt = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/atlas/NMT_inblockface.nii.gz')
    blikef = ants.image_read(fluor_CONFIG['output_dir']+'/reg3D/T1wlikeB_c.nii.gz')
    if isAffine:
        t = ants.registration(b, b_, type_of_transform='Affine')
        atlas_ = ants.apply_transforms(b, atlas, t['fwdtransforms'],'multiLabel')
        atlas1_ = ants.apply_transforms(b, atlas1, t['fwdtransforms'], 'multiLabel')
        atlas2_ = ants.apply_transforms(b, atlas2, t['fwdtransforms'], 'multiLabel')
        atlas4_ = ants.apply_transforms(b, atlas4, t['fwdtransforms'], 'multiLabel')
        nmt_ = ants.apply_transforms(b, nmt, t['fwdtransforms'], 'bSpline')
        atlas6_ = ants.apply_transforms(b, atlas6, t['fwdtransforms'], 'multiLabel')
        atlas7_ = ants.apply_transforms(b, atlas7, t['fwdtransforms'], 'multiLabel')
        blikef_ = ants.apply_transforms(b, blikef, t['fwdtransforms'], 'bSpline')
    else:
        atlas_ = ants.apply_transforms(b, atlas, [fluor_CONFIG['output_dir']+'/reg3D/xfms/b_regt1.mat'],'multiLabel',whichtoinvert=[True])
        atlas1_ = ants.apply_transforms(b, atlas1, [fluor_CONFIG['output_dir']+'/reg3D/xfms/b_regt1.mat'], 'multiLabel', whichtoinvert=[True])
        atlas2_ = ants.apply_transforms(b, atlas2, [fluor_CONFIG['output_dir']+'/reg3D/xfms/b_regt1.mat'], 'multiLabel',whichtoinvert=[True])
        atlas4_ = ants.apply_transforms(b, atlas4, [fluor_CONFIG['output_dir']+'/reg3D/xfms/b_regt1.mat'], 'multiLabel', whichtoinvert=[True])
        nmt_ = ants.apply_transforms(b, nmt, [fluor_CONFIG['output_dir']+'/reg3D/xfms/b_regt1.mat'], 'bSpline', whichtoinvert=[True])
        atlas6_ = ants.apply_transforms(b, atlas6, [fluor_CONFIG['output_dir']+'/reg3D/xfms/b_regt1.mat'], 'multiLabel', whichtoinvert=[True])
        atlas7_ = ants.apply_transforms(b, atlas7, [fluor_CONFIG['output_dir'] + '/reg3D/xfms/b_regt1.mat'],'multiLabel', whichtoinvert=[True])
        blikef_ = ants.apply_transforms(b, blikef, [fluor_CONFIG['output_dir']+'/reg3D/xfms/b_regt1.mat'], 'bSpline', whichtoinvert=[True])
    atlas_.to_file(fluor_CONFIG['output_dir']+'/blockface/atlas/D99_inOriginB.nii.gz')
    atlas1_ = ants.copy_image_info(b_origin, atlas1_)
    atlas1_.to_file(fluor_CONFIG['output_dir']+'/blockface/atlas/segmentation_inOriginB.nii.gz')
    atlas2_.to_file(fluor_CONFIG['output_dir']+'/blockface/atlas/SARM2_inOriginB.nii.gz')
    atlas4_ = ants.copy_image_info(b_origin, atlas4_)
    atlas4_.to_file(fluor_CONFIG['output_dir']+'/blockface/atlas/cerebellum_mask_inOriginB.nii.gz')
    nmt_=ants.copy_image_info(b_origin,nmt_)
    nmt_.to_file(fluor_CONFIG['output_dir']+'/blockface/atlas/TMP_inOriginB.nii.gz')
    atlas6_.to_file(fluor_CONFIG['output_dir']+'/blockface/atlas/CHARM1_inOriginB.nii.gz')
    atlas7_.to_file(fluor_CONFIG['output_dir'] + '/blockface/atlas/segmentation_edit_inOriginB.nii.gz')
    blikef_ = ants.copy_image_info(b_origin, blikef_)
    blikef_.to_file(fluor_CONFIG['output_dir']+'/blockface/atlas/T1wlikeB_inOriginB.nii.gz')



def repair_blockface():
    logger = loggerz.get_logger()
    logger.info('repair blockface with segmentation')
    b=ants.image_read(fluor_CONFIG['output_dir']+'/blockface/b_recon_oc_scale_clahe.nii.gz')
    seg = ants.image_read(fluor_CONFIG['output_dir']+'/blockface/atlas/segmentation_inOriginB.nii.gz')
    cere_mask = ants.image_read(fluor_CONFIG['output_dir']+'/blockface/atlas/cerebellum_mask_inOriginB.nii.gz')
    b_rmc=b-ants.mask_image(b,cere_mask,1)
    b_rmc_mask=ants.get_mask(b_rmc)
    b_rmc_mask=ants.morphology(b_rmc_mask,'erode',2)
    b_rmc_mask = ants.morphology(b_rmc_mask, 'dilate',2)
    b_rmc=ants.mask_image(b_rmc,b_rmc_mask,1)
    b_rmc.to_file(fluor_CONFIG['output_dir']+'/blockface/b_recon_oc_scale_rmc.nii.gz')
    seg = ants.morphology(seg, 'erode', 1)
    seg_data=seg.numpy()
    seg[:,:,:]=seg_data
    b_rmc_repair=b_rmc-ants.mask_image(b_rmc,seg,[1,5])
    b_rmc_repair_mask = ants.get_mask(b_rmc_repair)
    b_rmc_repair_mask=ants.morphology(b_rmc_repair_mask,'erode',3)
    b_rmc_repair_mask = ants.morphology(b_rmc_repair_mask, 'dilate',3)
    b_rmc_repair = ants.mask_image(b_rmc_repair, b_rmc_repair_mask, 1)
    b_rmc_repair.to_file(fluor_CONFIG['output_dir']+'/blockface/b_recon_oc_scale_rmc_repair.nii.gz')

def repair_seg_inBlockface():
    img = ants.image_read(fluor_CONFIG['output_dir'] + '/blockface/atlas/segmentation_edit_inOriginB.nii.gz')
    b=ants.image_read(fluor_CONFIG['output_dir']+'/blockface/b_recon_oc_scale_rmc_repair.nii.gz')
    mask=ants.get_mask(b)
    img=ants.copy_image_info(b,img)
    img_=ants.mask_image(img,mask)
    img_data=img_.numpy()
    index=None
    for i in range(img.shape[1]-1,0,-1):
        slice=img_data[:,i,:]
        if set(np.unique(slice).astype(int)) == {0,160, 210, 100}:
            print(i)
            index=i
            print(np.unique(slice).astype(int))
            break
    for ii in range(index,img.shape[1]):
        slice = img_data[:, ii, :]
        if 160 in np.unique(slice).astype(int) and 210 in np.unique(slice).astype(int):
            remaining_values = np.setdiff1d(np.unique(slice).astype(int), np.array([0,160, 210, 100]))
            print(remaining_values)
            tmp=img_data[:, ii, :]
            tmp[tmp==150]=100
            tmp[tmp == 200] = 100
    index=None
    for i in range(img.shape[1]-1,0,-1):
        slice=img_data[:,i,:]
        if set(np.unique(slice).astype(int)) == {0,2}:
            print(i)
            index=i
            print(np.unique(slice).astype(int))
            break
    for ii in range(index,img.shape[1]):
        slice = img_data[:, ii, :]
        if 2 in np.unique(slice).astype(int) :
            remaining_values = np.setdiff1d(np.unique(slice).astype(int), np.array([0,2]))
            print(remaining_values)
            tmp=img_data[:, ii, :]
            for t in remaining_values:
                tmp[tmp==t]=2
    index=None
    for i in range(0,img.shape[1]):
        slice=img_data[:,i,:]
        if set(np.unique(slice).astype(int)) == {0,2}:
            print(i)
            index=i
            print(np.unique(slice).astype(int))
            break
    for ii in range(index,0,-1):
        slice = img_data[:, ii, :]
        if 2 in np.unique(slice).astype(int) :
            remaining_values = np.setdiff1d(np.unique(slice).astype(int), np.array([0,2]))
            print(remaining_values)
            tmp=img_data[:, ii, :]
            for t in remaining_values:
                tmp[tmp==t]=2
    index = None
    for i in range(0, img.shape[1]):
        slice = img_data[:, i, :]
        if set(np.unique(slice).astype(int)) == {0, 2}:
            print(i)
            index = i
            print(np.unique(slice).astype(int))
            break
    count = 0
    for ii in range(index - 1, 0, -1):
        slice_origin = np.rot90(img_data[:, ii, :].copy()).copy()
        slice = slice_origin.copy()
        slice[slice > 0] = 100
        slice = touint8(slice)
        w = get_maskBywatershed(slice)
        for i in np.unique(w):
            area = w[w == i]
            if len(area) < 300:
                w[w == i] = 1
        # plot_show(slice[:, :], w, True)
        c_list = []
        for iii in np.unique(w):
            if iii != 1:
                w_ = w.copy()
                w_[w != iii] = 0
                sx, sy = centerxy_img(w_ * 35)
                if not (sx == 0 and sy == 0):
                    c_list.append((sx, sy))
        if len(c_list) == 3:
            max_y_point = max(c_list, key=lambda point: point[1])
            w[w != w[max_y_point[1], max_y_point[0]]] = 0
            w[w > 0] = 1
            slice_ = slice * w
            slice_[slice_ > 0] = 88
            img_data[:, ii, :] = np.rot90(slice_origin - slice_origin * w + slice_, 3).copy()
        else:
            count = count + 1
        if count > 10:
            break

    img[:, :, :] = img_data
    img.to_file(fluor_CONFIG['output_dir']+'/blockface/atlas/segmentation_edit_inOriginB_.nii.gz')