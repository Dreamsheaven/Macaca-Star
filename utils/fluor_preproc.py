import os
import ants
import cv2
import numpy as np
import yaml
from skimage import util
import albumentations as A
from utils.util import touint8
from utils.util_fluor import get_fslice_mask, plot_show, normalization, get_maskBywatershed, syn_toB_bySeg
from visdom import Visdom

fluor_YAML_PATH = os.getcwd() + '/config/fluor_sections_config.yaml'
fluor_CONFIG = yaml.safe_load(open(fluor_YAML_PATH, 'r'))

def repaire_blikefluo():
    is_ZBackground=True
    f = ants.image_read(fluor_CONFIG['output_dir']+'/fluor.nii.gz')
    tf = ants.image_read(fluor_CONFIG['output_dir']+'/fluor/blike_f.nii.gz')
    f=ants.iMath_normalize(f)*255
    tf = ants.iMath_normalize(tf) * 255
    tf_ = ants.from_numpy(np.zeros((f.shape[0],f.shape[1],f.shape[2])))
    tf_.set_spacing(f.spacing)
    tf_.set_direction(f.direction)
    tf_.set_origin(f.origin)
    tf_aug=ants.image_clone(tf_)
    CLAHE = A.CLAHE(clip_limit=(1.0, 2.0), tile_grid_size=(2, 2), always_apply=False, p=0.5)
    CLAHE2 = A.CLAHE(clip_limit=(1.0, 2.0), tile_grid_size=(10, 10), always_apply=False, p=0.5)
    CLAHE3 = A.CLAHE(clip_limit=(1.0, 2.0), tile_grid_size=(15, 15), always_apply=False, p=0.5)
    for i in range(0, f.shape[1]):
        print(str(i) + ' ' + str(f.shape[1]))
        f_slice_data = f[:, i, :].numpy()
        tf_slice_data=tf[:, i, :].numpy()
        tf_slice_data = tf_slice_data.astype(np.uint8)
        f_slice_data = f_slice_data.astype(np.uint8)
        tf_slice = ants.from_numpy(tf_slice_data)
        f_slice = ants.from_numpy(f_slice_data)
        #########################################################
        mask_slice= ants.image_clone(f_slice)
        mask_slice[:,:]=0
        inf=util.invert(f_slice[:,:].numpy())
        inf=cv2.GaussianBlur(inf, (3, 3), 1.5)
        inf = cv2.medianBlur(inf, 3)
        mask2=get_fslice_mask(f_slice.numpy().copy(),1000)
        result = ants.registration(f_slice, tf_slice, 'SyN',syn_sampling=32,
                                   reg_iterations=(10, 5, 0),flow_sigma=5)
        tf_slice = ants.apply_transforms(f_slice, tf_slice, result['fwdtransforms'])
        plot_show(f_slice.numpy(),mask2,False)
        tf_slice_=tf_slice
        tftmp=tf_slice_.numpy()
        tf_augtmp=normalization(tftmp.copy()*(1+normalization(inf)*0.3)*mask2[:, :])*255
        if is_ZBackground:
            maskb=f_slice.numpy().copy()
            maskb[maskb>0]=1
            tf_augtmp=tf_augtmp*maskb[:,:]
            kernel = np.ones((2, 2), np.uint8)
            tf_augtmp = cv2.morphologyEx(tf_augtmp, cv2.MORPH_CLOSE, kernel)*mask2

        tf_augtmp = touint8(tf_augtmp)
        tf_augtmp = CLAHE.apply(tf_augtmp, clip_limit=2.0)
        tf_augtmp = CLAHE2.apply(tf_augtmp, clip_limit=0.5)
        tf_augtmp = CLAHE3.apply(tf_augtmp, clip_limit=0.3)
        tf_[:, i, :]=tf_slice_.numpy()
        tf_aug[:, i, :] = tf_augtmp.copy()
    ants.image_write(tf_aug,fluor_CONFIG['output_dir']+'/fluor/blikef_repair2D_aug.nii.gz')


def fluor_SyNtoB_bySeg():
    isPlot=False
    viz = Visdom(env='slice2D_fluo_affinetoB')
    b = ants.image_read(fluor_CONFIG['output_dir']+'/blockface/b_recon_oc_scale_rmc_repair.nii.gz')
    tf = ants.image_read(fluor_CONFIG['output_dir']+ '/fluor/blikef_repair2D_aug.nii.gz')
    b_seg = ants.image_read(fluor_CONFIG['output_dir'] + '/blockface/atlas/segmentation_edit_inOriginB_.nii.gz')
    tf_=ants.new_image_like(tf,tf.numpy())
    tf_[:, :, :] = 0
    for index in range(0, tf.shape[1]):
    # for index in range(59,60):
        print(index)
        tf_[:,index,:]=0
        bslice = touint8(b.numpy()[:, index, :].copy())
        tfslice = touint8(tf.numpy()[:, index, :].copy())
        bslice = np.rot90(bslice).copy()
        tfslice = np.rot90(tfslice).copy()
        bsliceimg = ants.from_numpy(bslice)
        tfsliceimg = ants.from_numpy(tfslice)
        result1 = ants.registration(bsliceimg, tfsliceimg, 'Similarity', aff_metric='GC',outprefix=fluor_CONFIG['output_dir']+ '/reg2D/xfms/ftob_affine_iter1_'+str(index)+'_')
        tfsliceimg = ants.apply_transforms(bsliceimg, tfsliceimg, result1['fwdtransforms'],'linear')
        tfslice = tfsliceimg.numpy().copy()
        bmask = bslice.copy()
        tfmask = tfslice.copy()
        bmask[bmask > 50] = 100
        tfmask[tfmask > 50] = 100
        bslice_mask = get_maskBywatershed(bmask)
        tfslice_mask = get_maskBywatershed(tfmask)
        bslice_mask[bslice_mask == 0] = 1
        tfslice_mask[tfslice_mask == 0] = 1
        plot_show(tfslice_mask, bslice_mask, isPlot)
        newbf_data=syn_toB_bySeg(bsliceimg,bslice_mask,tfsliceimg,tfslice_mask,np.rot90(b_seg[:,index,:].numpy()),index)
        tf_[:, index, :]=np.rot90(newbf_data.copy(),3)
        viz.image(bslice[:, :], win='1')
        viz.image(newbf_data[:, :], win='2')
        viz.image(np.rot90(touint8(tf.numpy()[:, index, :].copy())), win='3')
    tf_.to_file(fluor_CONFIG['output_dir']+ '/reg2D/blikef_affine.nii.gz')



