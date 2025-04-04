import os
import shutil

import albumentations as A
import ants
import utils.Logger as loggerz
from utils.util import touint8
import yaml


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
