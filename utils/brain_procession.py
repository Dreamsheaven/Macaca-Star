import os
import shutil
import time
import SimpleITK as sitk
import albumentations as A
import ants
import cv2 as cv
import nibabel as nib
import numpy as np
import scipy.signal
import tifffile
import yaml

import utils.Logger as loggerz

LOGGER = None
ABP_N4 = False
YAML_PATH = os.getcwd() + '/config/config.yaml'
CONFIG = yaml.safe_load(open(YAML_PATH, 'r'))
SUBJECT_DIR = CONFIG['subject_dir']
SCALE = CONFIG['resolution']


# SUBJECT_DIR='/mnt/f4cb1796-fbd8-42b0-8743-1e8ebdb9f1d9/macaque/zzb/6.15/pipeline3'

def init_logger():
    global LOGGER
    LOGGER = logger = loggerz.LOGGER
    LOGGER.info('START T1<-->NMT')


def seg_white_gray():
    img = ants.image_read(SUBJECT_DIR + '/T1w_bc.nii.gz')
    mask = ants.get_mask(img, low_thresh=None, high_thresh=None, cleanup=2)
    ants.image_write(mask, SUBJECT_DIR + '/seg/mask.nii.gz')
    w = ants.atropos(img, mask, i='Kmeans[2]', m='[0.2,1x1x1]', priorweight=0.5, c='[20,0]')
    ants.image_write(w['segmentation'], SUBJECT_DIR + '/seg/seg.nii.gz')
    for i in range(0, len(w['probabilityimages'])):
        ants.image_write(w['probabilityimages'][i], SUBJECT_DIR + '/seg/seg' + str(i) + '.nii.gz')


def brain_orientation_correction():
    LOGGER.info('MRI orientation correction')
    oc_by_metrix = CONFIG['MRI_oc_bymetrix']
    if os.path.exists(SUBJECT_DIR + '/T1w.nii.gz'):
        t1 = ants.image_read(SUBJECT_DIR + '/T1w.nii.gz')
    elif os.path.exists(SUBJECT_DIR + '/T1w.nii'):
        t1 = ants.image_read(SUBJECT_DIR + '/T1w.nii')
    # t1w_bc = bias_correction(t1w_path, subject_dir)
    if not os.path.exists(SUBJECT_DIR + '/T1_NMT'):
        os.makedirs(SUBJECT_DIR + '/T1_NMT')
    # affine_metrix = np.zeros((3, 3))
    # affine_metrix[1, 0] = -0.5
    # affine_metrix[2, 1] = -0.5
    # affine_metrix[0, 2] = 0.5
    if oc_by_metrix:
        affine_metrix1 = np.array([[0, 0, 0.5],
                                   [-0.5, 0.5, 0],
                                   [0, -0.5, 0]])

        # affine_metrix4 = np.array([[0, 0, 0.5],
        #                           [0, 0.5, 0],
        #                           [0.5, 0, 0]])
        t1.set_direction(affine_metrix1)
        if not os.path.exists(SUBJECT_DIR + '/T1_NMT/tmp'):
            os.makedirs(SUBJECT_DIR + '/T1_NMT/tmp')
        ants.image_write(t1, SUBJECT_DIR + '/T1_NMT/tmp/T1w_oc.nii.gz')
    else:
        nmt = ants.image_read(
            os.getcwd() + '/template/NMT/0.25mm/NMT_v2.0_sym_fh.nii.gz')
        transform = ants.registration(nmt, t1, type_of_transform='Similarity')
        t1_rigid = ants.apply_transforms(nmt, t1, transform['fwdtransforms'], 'bSpline')
        ants.image_write(t1_rigid,SUBJECT_DIR + '/T1_NMT/T1w_oc.nii.gz')

def affine(type='T1w.nii.gz'):
    outprefix = None
    output = None
    if type == 'T1PI_0.25mm.nii.gz':
        nmt = ants.image_read(
            SUBJECT_DIR + '/PI_T1/tmp/PI_0.25mm_clahe_denoise.nii.gz')
        t1 = ants.image_read(SUBJECT_DIR + '/PI_T1/T1PI_0.25mm.nii.gz')
        if not os.path.exists(SUBJECT_DIR + '/PI_T1/xfms'):
            os.makedirs(SUBJECT_DIR + '/PI_T1/xfms')
        outprefix = SUBJECT_DIR + '/PI_T1/xfms/T1PI_to_PI_affine'
        output = SUBJECT_DIR + '/PI_T1/tmp/T1PI_affine_to_PI.nii.gz'
    elif type == 'PI_0.25mm_clahe_denoise.nii.gz':
        nmt = ants.image_read(
            os.getcwd() + '/template/NMT/0.25mm/NMT_v2.0_sym_SS.nii.gz')
        t1 = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_0.25mm_clahe_denoise.nii.gz')
        if not os.path.exists(SUBJECT_DIR + '/PI_T1/xfms'):
            os.makedirs(SUBJECT_DIR + '/PI_T1/xfms')
        outprefix = SUBJECT_DIR + '/PI_T1/xfms/PI_to_NMT_affine'
        output = SUBJECT_DIR + '/PI_T1/PI_0.25mm_clahe_denoise.nii.gz'
    else:
        # LOGGER.info('MRI to NMT with affine')
        nmt = ants.image_read(
            os.getcwd() + '/template/NMT/0.25mm/NMT_v2.0_sym.nii.gz')
        t1 = ants.image_read(SUBJECT_DIR + '/T1_NMT/tmp/T1w_oc.nii.gz')
        if not os.path.exists(SUBJECT_DIR + '/T1_NMT/xfms'):
            os.makedirs(SUBJECT_DIR + '/T1_NMT/xfms')
        outprefix = SUBJECT_DIR + '/T1_NMT/xfms/T1_to_NMT_affine'
        output = SUBJECT_DIR + '/T1_NMT/T1w_oc_affine.nii.gz'
    t1_affine_transform = ants.registration(nmt, t1, type_of_transform='TRSAA', outprefix=outprefix)
    t1_affine = ants.apply_transforms(nmt, t1, t1_affine_transform['fwdtransforms'], 'bSpline')
    ants.image_write(t1_affine, output)
    # LOGGER.info('Save to ' + SUBJECT_DIR + '/T1_NMT/T1w_oc_affine.nii.gz')


def nonlinear(type='SyN', image='T1w_oc.nii.gz'):
    LOGGER.info('Nonlinear to NMT')
    nmt = None
    brain = ''
    t1 = None
    outprefix = None
    if image == 'T1w_brain_bc.nii.gz':
        nmt = ants.image_read(
            os.getcwd() + '/template/NMT/0.25mm/NMT_v2.0_sym_05mm_SS.nii.gz')
        brain = 'brain'
        t1 = ants.image_read(SUBJECT_DIR + '/T1_NMT/' + image)
        outprefix = SUBJECT_DIR + '/T1_NMT/xfms/T1_to_NMT_' + brain + '_'
        output = SUBJECT_DIR + '/T1_NMT/nonlinear/T1_' + brain + '.nii.gz'
    elif image == 'T1PI_0.25mm.nii.gz':
        nmt = ants.image_read(
            SUBJECT_DIR + '/PI_T1/PI_brain_0.25mm.nii.gz')
        brain = 'correction'
        t1 = ants.image_read(SUBJECT_DIR + '/PI_T1/T1PI_0.25mm.nii.gz')
        if not os.path.exists(SUBJECT_DIR + '/PI_T1/nonlinear'):
            os.makedirs(SUBJECT_DIR + '/PI_T1/nonlinear')
        if not os.path.exists(SUBJECT_DIR + '/PI_T1/xfms'):
            os.makedirs(SUBJECT_DIR + '/PI_T1/xfms')
        outprefix = SUBJECT_DIR + '/PI_T1/xfms/T1PI_to_PI_' + brain + '_'
        output = SUBJECT_DIR + '/PI_T1/T1PI_' + 'correction' + '.nii.gz'
    else:
        # LOGGER.info('MRI to NMT with nonlinear')
        nmt = ants.image_read(
            os.getcwd() + '/template/NMT/0.25mm/NMT_v2.0_sym_fh.nii.gz')
        if not os.path.exists(SUBJECT_DIR + '/T1_NMT/nonlinear'):
            os.makedirs(SUBJECT_DIR + '/T1_NMT/nonlinear')
        brain = 'nonlinear'
        t1 = ants.image_read(SUBJECT_DIR + '/T1_NMT/' + image)
        outprefix = SUBJECT_DIR + '/T1_NMT/xfms/T1_to_NMT_' + brain + '_'
        output = SUBJECT_DIR + '/T1_NMT/nonlinear/T1_' + brain + '.nii.gz'

    t1_nonlinear_transform = ants.registration(nmt, t1, type_of_transform=type,
                                               reg_iterations=(40, 20, 0), multivariate_extras=(
            ('mattes', nmt, t1, 0.5, 0), ('demons', nmt, t1, 0.2, 0), ('MeanSquares', nmt, t1, 0.3, 0)),
                                               outprefix=outprefix)
    t1_nonlinear = ants.apply_transforms(nmt, t1, t1_nonlinear_transform['fwdtransforms'], 'bSpline')
    ants.image_write(t1_nonlinear, output)


def make_brain_mask():
    # LOGGER.info('skull brain')
    nmt_mask = ants.image_read(
        os.getcwd() + '/template/NMT/0.25mm/NMT_v2.0_asym_fh_brainmask.nii.gz')
    t1 = ants.image_read(SUBJECT_DIR + '/T1_NMT/T1w_oc.nii.gz')
    t1_mask = ants.apply_transforms(t1, nmt_mask, SUBJECT_DIR + '/T1_NMT/xfms/T1_to_NMT_nonlinear_1InverseWarp.nii.gz',
                                    'multiLabel')
    t1_mask = ants.apply_transforms(t1, t1_mask, SUBJECT_DIR + '/T1_NMT/xfms/T1_to_NMT_nonlinear_0GenericAffine.mat',
                                    'multiLabel',
                                    whichtoinvert=[True])
    ants.image_write(t1_mask, SUBJECT_DIR + '/T1_NMT/T1_brain_mask.nii.gz')
    t1_brain = ants.mask_image(t1, t1_mask)
    ants.image_write(t1_brain, SUBJECT_DIR + '/T1_NMT/tmp/T1_brain_affine.nii.gz')


def bias_correction():
    LOGGER.info('bias correction')
    image = ants.image_read(SUBJECT_DIR + '/T1_NMT/tmp/T1_brain_affine.nii.gz')
    # t1w_norm=ants.iMath(t1w,'Normalize')
    if not ABP_N4:
        t1w_bc = ants.n4_bias_field_correction(image, convergence={"iters": [50, 50, 50, 50], "tol": 1e-7},
                                               return_bias_field=False)
    else:
        t1w_bc = ants.abp_n4(image, intensity_truncation=(0.00, 0.95, image.max()))
    ants.image_write(t1w_bc, SUBJECT_DIR + '/T1_NMT/T1w_brain_bc.nii.gz')


def make_brain_atlas():
    #  LOGGER.info('atlas to subject space')
    nonlinear('ElasticSyN', 'T1w_brain_bc.nii.gz')
    nmt_atlas = ants.image_read(
        os.getcwd() + '/template/NMT/0.5mm/D99_atlas_in_NMT_v2.0_sym_05mm.nii.gz')
    t1 = ants.image_read(SUBJECT_DIR + '/T1_NMT/T1w_brain_bc.nii.gz')
    t1_atlas = ants.apply_transforms(t1, nmt_atlas, SUBJECT_DIR + '/T1_NMT/xfms/T1_to_NMT_brain_1InverseWarp.nii.gz',
                                     'multiLabel')
    t1_atlas = ants.apply_transforms(t1, t1_atlas, SUBJECT_DIR + '/T1_NMT/xfms/T1_to_NMT_brain_0GenericAffine.mat',
                                     'multiLabel', whichtoinvert=[True])
    if not os.path.exists(SUBJECT_DIR + '/T1_NMT/atlas'):
        os.makedirs(SUBJECT_DIR + '/T1_NMT/atlas')
    ants.image_write(t1_atlas, SUBJECT_DIR + '/T1_NMT/atlas/D99_in_subject.nii.gz')

    nmt_atlas = ants.image_read(
        os.getcwd() + '/template/NMT/0.5mm/supplemental_SARM/SARM_6_in_NMT_v2.0_sym_05mm.nii.gz')
    t1 = ants.image_read(SUBJECT_DIR + '/T1_NMT/T1w_brain_bc.nii.gz')
    t1_atlas = ants.apply_transforms(t1, nmt_atlas, SUBJECT_DIR + '/T1_NMT/xfms/T1_to_NMT_brain_1InverseWarp.nii.gz',
                                     'multiLabel')
    t1_atlas = ants.apply_transforms(t1, t1_atlas, SUBJECT_DIR + '/T1_NMT/xfms/T1_to_NMT_brain_0GenericAffine.mat',
                                     'multiLabel', whichtoinvert=[True])
    if not os.path.exists(SUBJECT_DIR + '/T1_NMT/atlas'):
        os.makedirs(SUBJECT_DIR + '/T1_NMT/atlas')
    ants.image_write(t1_atlas, SUBJECT_DIR + '/T1_NMT/atlas/SARM_in_subject.nii.gz')


def crop_brain(LR):
    img_data = ants.image_read(SUBJECT_DIR + '/T1_NMT/T1w_brain_bc_dn.nii.gz')
    # atlas_data = ants.image_read(os.getcwd()+'/template/D99_atlas_in_NMT_v2.0_sym_L.nii.gz')
    # brainmask_data = ants.image_read(SUBJECT_DIR + '/T1_NMT/T1_brain_mask.nii.gz')
    if LR == 'L':
        img_data_ = img_data[int(img_data.shape[0] / 2):img_data.shape[0], 0:img_data.shape[1], 0:img_data.shape[2]]
        # atlas_data_ = atlas_data[int(atlas_data.shape[0] / 2):atlas_data.shape[0], 0:atlas_data.shape[1],
        #               0:atlas_data.shape[2]]
        # brainmask_data_ = brainmask_data[int(brainmask_data.shape[0] / 2):brainmask_data.shape[0], 0:brainmask_data.shape[1],
        #               0:brainmask_data.shape[2]]
    elif LR == 'R':
        img_data_ = img_data[0:int(img_data.shape[0] / 2), 0:img_data.shape[1], 0:img_data.shape[2]]
        # atlas_data_ = atlas_data[0:int(atlas_data.shape[0] / 2), 0:atlas_data.shape[1], 0:atlas_data.shape[2]]
        # brainmask_data_ = brainmask_data[0:int(brainmask_data.shape[0] / 2), 0:brainmask_data.shape[1], 0:brainmask_data.shape[2]]
    img_data_LR = ants.from_numpy(img_data_, origin=img_data.origin, spacing=img_data.spacing,
                                  direction=img_data.direction)
    # atlas_data_LR = ants.from_numpy(atlas_data_, origin=atlas_data.origin, spacing=atlas_data.spacing,
    #                                 direction=atlas_data.direction)
    # brainmask_data_LR = ants.from_numpy(brainmask_data_, origin=brainmask_data.origin, spacing=brainmask_data.spacing,
    #                                 direction=brainmask_data.direction)
    if not os.path.exists(SUBJECT_DIR + '/PI_T1'):
        os.makedirs(SUBJECT_DIR + '/PI_T1')
    ants.image_write(img_data_LR, SUBJECT_DIR + '/PI_T1/T1w_brain_bc_' + LR + '.nii.gz')
    if not os.path.exists(SUBJECT_DIR + '/PI_T1/atlas'):
        os.makedirs(SUBJECT_DIR + '/PI_T1/atlas')
    # ants.image_write(atlas_data_LR, SUBJECT_DIR + '/PI_T1/atlas/atlas_in_subject_' + LR + '.nii.gz')
    # ants.image_write(brainmask_data_LR, SUBJECT_DIR + '/PI_T1/T1_brainmask_' + LR + '.nii.gz')


def clahe_image():
    LOGGER.info('START PI preprocession')
    pi = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rm_norm_dn_' + str(SCALE) + 'mm.nii.gz')
    if CONFIG['clahe']:
        LOGGER.info('START PI clahe')
        CLAHE = A.CLAHE(clip_limit=(0.5, 2.0), tile_grid_size=(8, 8), always_apply=False, p=0.5)
        spacing = pi.spacing
        direction = pi.direction
        pi = pi[:, :, :]
        space = np.max(pi) - np.min(pi)
        norm = (pi - np.min(pi)) * 255 / space
        pi = norm.astype(np.uint8)
        for i in range(0, pi.shape[0]):
            pi_splice = pi[i, :, :]
            pi_splice = CLAHE.apply(pi_splice)
            pi[i, :, :] = pi_splice
        pi = pi.astype(np.uint8)
        pi_ = ants.from_numpy(pi)
        pi_.set_spacing(spacing)
        pi_.set_direction(direction)
    else:
        pi_ = pi
    ants.image_write(pi_, SUBJECT_DIR + '/PI_T1/tmp/PI_rm_norm_dn_ic_' + str(SCALE) + 'mm.nii.gz')

    #     clahe = cv.createCLAHE(clipLimit=100.0, tileGridSize=(10, 10))
    #     pi = tifffile.imread(SUBJECT_DIR + '/PI.tif')
    #     # pi=pi.astype(np.float32)
    #     for i in range(0, pi.shape[0]):
    #         # pi_splice = cv.cvtColor(pi[i, :, :], cv.COLOR_GRAY2BGR)
    #         pi_splice = pi[i, :, :]
    #         pi_splice = clahe.apply(pi_splice)
    #         pi[i, :, :] = pi_splice
    #     tifffile.imwrite(SUBJECT_DIR + '/PI_clahe.tif', pi)
    #     tif_to_nii(pi, 0.075)


def tif_to_nii(data, scale=0.5, name='clahe'):
    # LOGGER.info('tif to nifti')
    nii_header = nib.Nifti1Header()
    nii_header.set_xyzt_units('mm', 'sec')
    nii_header.set_data_dtype(np.uint16)  # 或者 np.float32，具体根据数据类型而定
    nii_header.set_data_shape(data.shape)
    x_scale = scale
    y_scale = scale
    z_scale = scale
    x_trans = 0.0
    y_trans = 0.0
    z_trans = 0.0
    affine_metrix = np.array([[x_scale, 0, 0, x_trans],
                              [0, y_scale, 0, y_trans],
                              [0, 0, z_scale, z_trans],
                              [0, 0, 0, 1]])
    nii_file = nib.Nifti1Image(data, affine_metrix, header=nii_header)
    if not os.path.exists(SUBJECT_DIR + '/PI_T1/tmp'):
        os.makedirs(SUBJECT_DIR + '/PI_T1/tmp')
    nib.save(nii_file, SUBJECT_DIR + '/PI_T1/tmp/PI_' + str(scale) + 'mm_' + name + '.nii.gz')


def tif_to_nii():
    # LOGGER.info('tif to nifti')
    scale = SCALE
    if not os.path.exists(SUBJECT_DIR + '/PI_T1/tmp'):
        os.makedirs(SUBJECT_DIR + '/PI_T1/tmp')

    if os.path.exists(SUBJECT_DIR + '/PI.tif'):
        pi = tifffile.imread(SUBJECT_DIR + '/PI.tif')
        img=ants.from_numpy(pi)
        img.set_spacing((scale,scale,scale))
        ants.image_write(img, SUBJECT_DIR + '/PI_T1/tmp/PI_' + str(scale) + 'mm' + '.nii.gz')
        # nii_header = nib.Nifti1Header()
        # nii_header.set_xyzt_units('mm', 'sec')
        # nii_header.set_data_dtype(np.uint16)  # 或者 np.float32，具体根据数据类型而定
        # nii_header.set_data_shape(pi.shape)
        # x_scale = scale
        # y_scale = scale
        # z_scale = scale
        # x_trans = 0.0
        # y_trans = 0.0
        # z_trans = 0.0
        # affine_metrix = np.array([[x_scale, 0, 0, x_trans],
        #                           [0, y_scale, 0, y_trans],
        #                           [0, 0, z_scale, z_trans],
        #                           [0, 0, 0, 1]])
        # nii_file = nib.Nifti1Image(pi, affine_metrix, header=nii_header)
        # nib.save(nii_file, SUBJECT_DIR + '/PI_T1/tmp/PI_' + str(scale) + 'mm' + '.nii.gz')
    elif os.path.exists(SUBJECT_DIR + '/PI.nii.gz'):
        shutil.copy(SUBJECT_DIR + '/PI.nii.gz', SUBJECT_DIR + '/PI_T1/tmp/PI_' + str(scale) + 'mm' + '.nii.gz')
    elif os.path.exists(SUBJECT_DIR + '/PI.nii'):
        shutil.copy(SUBJECT_DIR + '/PI.nii', SUBJECT_DIR + '/PI_T1/tmp/PI_' + str(scale) + 'mm' + '.nii.gz')


def denoise(iter=0):
    LOGGER.info('Denoise')

    if iter == 0:
        pi = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_' + str(SCALE) + 'mm.nii.gz')
        # pi_denoise = ants.denoise_image(pi, p=1, r=3, shrink_factor=1)
        # pi_denoise = ants.denoise_image(pi_denoise, p=1, r=3, shrink_factor=1)
        pi_denoise = ants.smooth_image(pi, sigma=1, max_kernel_width=5)
    elif iter == 1:
        pi = ants.image_read(SUBJECT_DIR + '/PI_T1/PI_rm_' + str(SCALE) + 'mm.nii.gz')
        # pi_denoise = ants.denoise_image(pi, p=1, r=3, shrink_factor=1)
        # pi_denoise = ants.denoise_image(pi_denoise, p=1, r=3, shrink_factor=1)
        pi_denoise = ants.smooth_image(pi, sigma=1, max_kernel_width=3)
    else:
        pi = ants.image_read(SUBJECT_DIR + '/PI_T1/PI_rm_' + str(SCALE) + 'mm.nii.gz')
        # pi_denoise = ants.smooth_image(pi, sigma=1, max_kernel_width=3)
        # pi_ic = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rm_ic_'+str(SCALE)+'mm.nii.gz')

        # pi_denoise = ants.denoise_image(pi, shrink_factor=1, p=1, r=3, noise_model="Rician")
        # pi_denoise = ants.denoise_image(pi_denoise, shrink_factor=1, p=1, r=3, noise_model="Gaussian")
        pi_denoise = ants.smooth_image(pi, sigma=1, max_kernel_width=2)
        # ants.image_write(pi_denoise, SUBJECT_DIR + '/PI_T1/tmp/PI_rm_ic_dn_' + str(SCALE) + 'mm.nii.gz')
    ants.image_write(pi_denoise, SUBJECT_DIR + '/PI_T1/tmp/PI_rm_dn_' + str(SCALE) + 'mm.nii.gz')


def fftt():
    pi = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_' + str(SCALE) + 'mm' + '.nii.gz')

    for i in range(0, pi.shape[0]):
        image = pi[i, :, :]  # 加载为灰度图像
        fft = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = 20 * np.log(np.abs(fft_shift))

        rows, cols = image.shape
        crow, ccol = int(rows / 2), int(cols / 2)  # 中心点坐标

        # 创建一个与频谱大小相同的掩模
        mask = np.ones((rows, cols), np.uint8)
        r = 4  # 指定带通滤波器的半径
        rr = 8

        # mask[0:crow-rr, ccol-r:ccol+r] = 0
        # mask[crow+rr:rows, ccol-r:ccol+r] = 0
        mask[crow - r:crow + r, 0:ccol - rr] = 0
        mask[crow - r:crow + r, ccol + rr:cols] = 0

        fft_shift_filtered = fft_shift * mask

        fft_filtered = np.fft.ifftshift(fft_shift_filtered)
        image_filtered = np.fft.ifft2(fft_filtered)
        # image_filtered = np.abs(image_filtered).astype(np.uint8)
        image_filtered = np.abs(image_filtered)
        pi[i, :, :] = image_filtered
    ants.image_write(pi, SUBJECT_DIR + '/PI_T1/PI_0.075mm_clahe_denoise.nii.gz')


def resample_image(iter=0):
    if iter == 0:
        LOGGER.info('START resample iter 0')
        pi = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rm_dn_' + str(SCALE) + 'mm.nii.gz')
        pi_25 = ants.resample_image(pi, (0.25, 0.25, 0.25), interp_type=4)
        pi_25_origin = pi_25
        # pi = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_' + str(SCALE) + 'mm.nii.gz')
        # pi_25 = ants.resample_image(pi, (0.25, 0.25, 0.25), interp_type=4)
        # pi_denoise = ants.denoise_image(pi_25, p=1, r=3)
        # pi_denoise = ants.denoise_image(pi_denoise, p=1, r=3)
        # pi_25=pi_denoise
        # ants.image_write(pi_denoise, SUBJECT_DIR + '/PI_T1/tmp/PI_rm_dn_' + str(SCALE) + 'mm.nii.gz')
    elif iter == 1:
        LOGGER.info('START resample iter 1')
        pi_ = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rm_dn_' + str(SCALE) + 'mm.nii.gz')
        pi = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rm_norm_dn_ic_' + str(SCALE) + 'mm.nii.gz')
        c_mask=ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/cerebellum_mask_' + str(SCALE) + 'mm.nii.gz')
        if CONFIG['resample_by_affine']:
            LOGGER.info('resample_by_affine')
            fix = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/NMT_v2.0_asym_brain_L.nii.gz')
            affine_transform = ants.registration(fix, pi_, type_of_transform='Affine', reg_iterations=(40, 20, 0),
                                                 outprefix=SUBJECT_DIR + '/PI_T1/xfms/PI_downsampleByAffine_')
            pi_25 = ants.apply_transforms(fix, pi, affine_transform['fwdtransforms'], 'bSpline')
            pi_25_origin = ants.apply_transforms(fix, pi_, affine_transform['fwdtransforms'], 'bSpline')
            c_mask_25 = ants.apply_transforms(fix, c_mask, affine_transform['fwdtransforms'], 'genericLabel')
        else:
            pi_25 = ants.resample_image(pi, (0.25, 0.25, 0.25), interp_type=4)
            affine_transform = ants.registration(pi_25, pi, type_of_transform='Affine', reg_iterations=(40, 20, 0),
                                                 outprefix=SUBJECT_DIR + '/PI_T1/xfms/PI_downsampleByAffine_')
            c_mask_25 = ants.apply_transforms(pi_25, c_mask, affine_transform['fwdtransforms'], 'genericLabel')
            pi_25_origin = pi_25
        if not os.path.exists(SUBJECT_DIR + '/PI_T1/atlas'):
            os.makedirs(SUBJECT_DIR + '/PI_T1/atlas')
        ants.image_write(c_mask_25,SUBJECT_DIR+'/PI_T1/atlas/cerebellum_mask_in_PI_0.25mm.nii.gz')

    elif iter==2:
        LOGGER.info('START resample iter 2')
        pi = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rm_dn_icxyz_' + str(SCALE) + 'mm.nii.gz')
        pi_ = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rm_dn_' + str(SCALE) + 'mm.nii.gz')
        if CONFIG['resample_by_affine']:
            LOGGER.info('resample_by_affine')
            fix = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/NMT_v2.0_asym_brain_L.nii.gz')
            affine_transform = ants.registration(fix, pi_, type_of_transform='TRSAA', reg_iterations=(40, 20, 0),
                                                 outprefix=SUBJECT_DIR + '/PI_T1/xfms/PI_downsampleByAffine_')
            pi_25 = ants.apply_transforms(fix, pi, affine_transform['fwdtransforms'], 'bSpline')
            pi_25_origin = ants.apply_transforms(fix, pi_, affine_transform['fwdtransforms'], 'bSpline')
        else:
            pi_25 = ants.resample_image(pi, (0.25, 0.25, 0.25), interp_type=4)
            pi_25_origin = pi_25


    ants.image_write(pi_25, SUBJECT_DIR + '/PI_T1/tmp/PI_rm_norm_ic_dn_0.25mm.nii.gz')
    ants.image_write(pi_25_origin, SUBJECT_DIR + '/PI_T1/tmp/PI_rm_dn_0.25mm.nii.gz')
    pi_5 = ants.resample_image(pi_25, (0.5, 0.5, 0.5), interp_type=4)
    ants.image_write(pi_5, SUBJECT_DIR + '/PI_T1/tmp/PI_rm_dn_0.5mm.nii.gz')


def resample_image_():
    LOGGER.info('START resample')
    pi = ants.image_read(SUBJECT_DIR + '/PI_T1/PI_0.075mm_clahe_denoise.nii.gz')
    pi_ = ants.resample_image(pi, (0.25, 0.25, 0.25), interp_type=4)
    ants.image_write(pi_, SUBJECT_DIR + '/PI_T1/tmp/PI_0.25mm_clahe_denoise.nii.gz')

    pi1 = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_0.075mm_nonclahe.nii.gz')
    # pi1_ = ants.resample_image(pi1, (0.25, 0.25, 0.25), interp_type=4)
    pi_to_pi_bc(pi_, pi1)


def pi_to_pi_bc(pi1, pi0):
    LOGGER.info('pi_to_pi_bc')
    pi_affine_transform = ants.registration(pi1, pi0, type_of_transform='Affine',
                                            reg_iterations=(40, 20, 0),
                                            outprefix=SUBJECT_DIR + '/PI_T1/xfms/PI_affine_to_PI_')
    pi_affine = ants.apply_transforms(pi1, pi0, pi_affine_transform['fwdtransforms'], 'bSpline')
    ants.image_write(pi_affine, SUBJECT_DIR + '/PI_T1/tmp/PI_nonclahe_0.25mm.nii.gz')


def make_pi_mask():
    LOGGER.info('make PI mask')
    bias_value = CONFIG['bias_value']
    print('bias_value:' + bias_value)
    label = None
    # img = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rm_norm_dn_bc_0.25mm.nii.gz')
    img = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rm_dn_0.25mm.nii.gz')
    # img_data = img[:, :, :]
    # mask = ants.get_mask(img, low_thresh=1200, cleanup=1)
    # ants.image_write(mask, SUBJECT_DIR + '/PI_T1/PI_mask.nii.gz')
    if bias_value == '2':
        mask = ants.get_mask(img, low_thresh=15)
        ants.image_write(mask, SUBJECT_DIR + '/PI_T1/PI_mask_0.25mm.nii.gz')
    elif bias_value == '3':
        mask = ants.get_mask(img, low_thresh=10, cleanup=1)
        w = ants.atropos(img, mask, i='Kmeans[' + bias_value + ']', m='[0.2,1x1x1]', priorweight=0.5, c='[5,0]')
        ants.image_write(w['segmentation'], SUBJECT_DIR + '/PI_T1/tmp/seg.nii.gz')
        lable1 = w['probabilityimages'][1]
        lable2 = w['probabilityimages'][2]
        label = lable1[:, :, :] + lable2[:, :, :]
        label12 = ants.from_numpy(label)
        label12.set_spacing(img.spacing)
        label12.set_direction(img.direction)
        label12 = ants.get_mask(label12, low_thresh=0.5, cleanup=1)
        label12.set_spacing(img.spacing)
        label12.set_direction(img.direction)
        label12.set_origin(img.origin)
        ants.image_write(label12, SUBJECT_DIR + '/PI_T1/PI_mask_0.25mm.nii.gz')
    elif bias_value == '4':
        mask = ants.get_mask(img, low_thresh=10, cleanup=1)
        w = ants.atropos(img, mask, i='Kmeans[' + bias_value + ']', m='[0.2,1x1x1]', priorweight=0.5, c='[5,0]')
        ants.image_write(w['segmentation'], SUBJECT_DIR + '/PI_T1/tmp/seg.nii.gz')
        lable1 = w['probabilityimages'][1]
        lable2 = w['probabilityimages'][2]
        lable3 = w['probabilityimages'][3]
        label = lable1[:, :, :] + lable2[:, :, :] + lable3[:, :, :]
        label12 = ants.from_numpy(label)
        label12.set_spacing(img.spacing)
        label12.set_direction(img.direction)
        label12 = ants.get_mask(label12, low_thresh=0.5, cleanup=1)
        label12.set_spacing(img.spacing)
        label12.set_direction(img.direction)
        label12.set_origin(img.origin)
        ants.image_write(label12, SUBJECT_DIR + '/PI_T1/PI_mask_0.25mm.nii.gz')
    # seg12 = ants.mask_image(img, label12)
    # ants.image_write(seg12, SUBJECT_DIR + '/PI_T1/tmp/PI_nonclahe_0.25mm_.nii.gz')


def mas_pi():
    LOGGER.info('exract PI brain')
    pi_mask = ants.image_read(SUBJECT_DIR + '/PI_T1/PI_mask_0.25mm.nii.gz')

    pi = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rm_norm_ic_dn_0.25mm.nii.gz')

    pi_brain = ants.mask_image(pi, pi_mask)
    ants.image_write(pi_brain, SUBJECT_DIR + '/PI_T1/tmp/PI_brain_0.25mm.nii.gz')


def mask_affine():
    LOGGER.info('create 0.075mm PI mask')
    mask = ants.image_read(SUBJECT_DIR + '/PI_T1/PI_mask_0.25mm.nii.gz')
    move = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rm_norm_dn_0.25mm.nii.gz')
    fix = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rm_dn_' + str(SCALE) + 'mm.nii.gz')
    affine_transforms = ants.registration(fix, move, type_of_transform='TRSAA',
                                          reg_iterations=(40, 20, 0), aff_metric='mattes',
                                          syn_metric='mattes')
    mask_ = ants.apply_transforms(fix, mask, affine_transforms['fwdtransforms'], 'genericLabel')
    ants.image_write(mask_, SUBJECT_DIR + '/PI_T1/tmp/PI_mask_' + str(SCALE) + 'mm.nii.gz')
    mask_affine2()


def mask_affine2():
    mask = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_mask_' + str(SCALE) + 'mm.nii.gz')
    img = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rm_dn_' + str(SCALE) + 'mm.nii.gz')
    pi_brain = ants.mask_image(img, mask)
    ants.image_write(pi_brain, SUBJECT_DIR + '/PI_T1/PI_rm_dn_' + str(SCALE) + 'mm.nii.gz')


def t1pi_correction():
    T1PI_correction = CONFIG['T1PI_correction']
    t1 = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/T1PI_tmp_0.25mm.nii.gz')
    t1_noc= ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/T1PI_nocerebellum_0.25mm.nii.gz')
    mask_c= ants.image_read(SUBJECT_DIR + '/PI_T1/atlas/cerebellum_mask_in_PI_0.25mm.nii.gz')
    # mask=ants.image_read(SUBJECT_DIR+'/PI_T1/PI_mask_0.25mm.nii.gz')
    t1=ants.mask_image(t1,mask_c)+t1_noc
    if not os.path.exists(SUBJECT_DIR + '/PI_T1/PI_brain_bc_icxyz_0.25mm.nii.gz'):
        output = SUBJECT_DIR + '/PI_T1/T1PI_' + 'correction_0.25mm' + '.nii.gz'
    else:
        output = SUBJECT_DIR + '/PI_T1/T1PI_' + 'correction_xyz_0.25mm' + '.nii.gz'

    if T1PI_correction:
        LOGGER.info('T1PI correction')
        if not os.path.exists(SUBJECT_DIR + '/PI_T1/PI_brain_bc_icxyz_0.25mm.nii.gz'):
            LOGGER.info('Loading PI_brain_bc_0.25mm.nii.gz')
            nmt = ants.image_read(
                SUBJECT_DIR + '/PI_T1/PI_brain_bc_0.25mm.nii.gz')
        else:
            LOGGER.info('Loading PI_brain_bc_icxyz_0.25mm.nii.gz')
            nmt = ants.image_read(
                SUBJECT_DIR + '/PI_T1/PI_brain_bc_icxyz_0.25mm.nii.gz')
        brain = 'correction'
        t1_ = t1.copy()
        nmt_ = nmt.copy()
        # t1_[:, 0:int(t1.shape[1] / 3), :] = 0
        # nmt_[:, 0:int(t1.shape[1] / 3), :] = 0

        if not os.path.exists(SUBJECT_DIR + '/PI_T1/nonlinear'):
            os.makedirs(SUBJECT_DIR + '/PI_T1/nonlinear')
        if not os.path.exists(SUBJECT_DIR + '/PI_T1/xfms'):
            os.makedirs(SUBJECT_DIR + '/PI_T1/xfms')
        outprefix = SUBJECT_DIR + '/PI_T1/xfms/T1PI_to_PI_' + brain + '_'

        t1_nonlinear_transform = ants.registration(nmt_, t1_, type_of_transform='SyN', syn_sampling=32,
                                                   reg_iterations=(20, 10, 0), multivariate_extras=(
                ('mattes',  nmt_,t1_, 1, 0)),outprefix=outprefix, flow_sigma=6)
        t1_nonlinear = ants.apply_transforms(nmt_, t1_, t1_nonlinear_transform['fwdtransforms'], 'bSpline')

        # t1_nonlinear_transform = ants.registration(nmt_, t1_nonlinear, type_of_transform='SyN', syn_sampling=32,
        #                                            reg_iterations=(20, 10, 0), multivariate_extras=(
        #         ('mattes', t1, nmt, 0.3, 32), ('demons', t1, nmt, 0.7, 32)),outprefix=outprefix, flow_sigma=3)
        # t1_nonlinear = ants.apply_transforms(nmt_, t1_nonlinear, t1_nonlinear_transform['fwdtransforms'], 'bSpline')
        # t1=t1_nonlinear
        # # t1[:, int(t1.shape[1] / 3):int(t1.shape[1]), :] = t1_nonlinear[:, int(t1.shape[1] / 3):int(t1.shape[1]), :]
        # t1_nonlinear = t1
    else:
        LOGGER.info('T1PI correction is false')
        t1_nonlinear = t1
    # t1_nonlinear=ants.mask_image(t1_nonlinear,mask)
    ants.image_write(t1_nonlinear, output)
    # further_c()


def make_occ_mask(img):
    mask = np.zeros((img.shape[0], img.shape[1], img.shape[2]))
    mask[15:int(0.8 * img.shape[0]), 0:int(0.3 * img.shape[1]), int(0.35 * img.shape[2]):img.shape[2]] = 1
    mask = ants.from_numpy(mask)
    mask.set_spacing([0.25, 0.25, 0.25])
    mask.set_direction(img.direction)
    return mask


def further_c():
    nmt = ants.image_read(
        SUBJECT_DIR + '/PI_T1/PI_brain_0.25mm.nii.gz')
    brain = 'correction'
    t1 = ants.image_read(SUBJECT_DIR + '/PI_T1/T1PI_correction.nii.gz')
    outprefix = SUBJECT_DIR + '/PI_T1/xfms/T1PI_to_PI_' + brain + '_'
    output = SUBJECT_DIR + '/PI_T1/T1PI_' + 'correction_' + '.nii.gz'
    t1_nonlinear_transform = ants.registration(nmt, t1, type_of_transform='Elastic', syn_sampling=32,
                                               syn_metric="mattes",
                                               reg_iterations=(100, 50, 10),
                                               outprefix=outprefix)
    t1_nonlinear = ants.apply_transforms(nmt, t1, t1_nonlinear_transform['fwdtransforms'], 'bSpline')
    ants.image_write(t1_nonlinear, output)


def subcor_atlas_to_t1pi():
    LOGGER.info('sub_atlas_to_t1pi')
    if not os.path.exists(SUBJECT_DIR + '/PI_T1/atlas'):
        os.makedirs(SUBJECT_DIR + '/PI_T1/atlas')
    t1pi2 = ants.image_read(SUBJECT_DIR + '/PI_T1/T1PI_correction_0.25mm.nii.gz')
    pi = ants.image_read(SUBJECT_DIR + '/PI_T1/PI_brain_bc_0.25mm.nii.gz')
    # t1pi2 = pi
    atlas = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/supplemental_SARM/SARM_6_in_NMT_v2.0_sym_L.nii.gz')
    atlas2 = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/supplemental_SARM/subcortex.nii.gz')
    t1pi_to_to_nmt_list = t1pi_to_t1_to_nmt(t1pi2)
    # t1pi_to_syn1 = t1pi_to_to_nmt_list[0]
    atlas_ = ants.apply_transforms(pi, atlas, t1pi_to_to_nmt_list[2], 'multiLabel')
    atlas2 = ants.apply_transforms(pi, atlas2, t1pi_to_to_nmt_list[2], 'multiLabel')
    ants.image_write(atlas_, SUBJECT_DIR + '/PI_T1/atlas/SARM_2_in_PI_0.25mm.nii.gz')
    ants.image_write(atlas2, SUBJECT_DIR + '/PI_T1/atlas/subcortex.nii.gz')


def subcor_atlas_to_t1pi2():
    LOGGER.info('atlas_to_t1pi')
    # T1 is as a bridge
    if not os.path.exists(SUBJECT_DIR + '/PI_T1/atlas'):
        os.makedirs(SUBJECT_DIR + '/PI_T1/atlas')
    nmt = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/NMT_v2.0_asym_brain_L.nii.gz')
    atlas2 = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/supplemental_SARM/subcortex.nii.gz')
    t1pi = ants.image_read(SUBJECT_DIR + '/PI_T1/T1PI_0.25mm.nii.gz')
    pi = ants.image_read(SUBJECT_DIR + '/PI_T1/PI_brain_bc_0.25mm.nii.gz')
    # atlas = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/D99_atlas_in_NMT_v2.0_sym_L_edit.nii.gz')
    atlas1 = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/supplemental_SARM/SARM_6_in_NMT_v2.0_sym_L.nii.gz')
    # seg = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/NMT_v2.0_sym_segmentation_L.nii.gz')
    if os.path.exists(SUBJECT_DIR + '/PI_T1/T1w_brain_bc_L.nii.gz'):
        LOGGER.warning('PI <-T1w-> NMT')
        t1 = ants.image_read(SUBJECT_DIR + '/PI_T1/T1w_brain_bc_L.nii.gz')
        pi_SyN_transform0 = ants.registration(t1, t1pi, type_of_transform='SyNAggro',
                                              reg_iterations=(80, 40, 20), flow_sigma=20, grad_step=0.4, aff_sampling=32,syn_sampling=64,
                                              multivariate_extras=(
                                                  ('mattes', t1, t1pi, 0.2, 32), ('MeanSquares', t1, t1pi, 0.8, 32)),
                                              outprefix=SUBJECT_DIR + '/PI_T1/xfms/PI_to_T1w_SyN_')
        t1pi_inT1 = ants.apply_transforms(t1, t1pi, pi_SyN_transform0['fwdtransforms'], 'bSpline')
        pi_inT1 = ants.apply_transforms(t1, pi, pi_SyN_transform0['fwdtransforms'], 'bSpline')

        # ants.image_write(pi_inT1, SUBJECT_DIR + '/PI_T1/atlas/PI_inT1_0.25mm.nii.gz')
        # ants.image_write(t1pi_inT1, SUBJECT_DIR + '/PI_T1/atlas/T1PI_inT1_0.25mm.nii.gz')

    else:
        LOGGER.warning('No T1 image; PI <--> NMT')
        pi_inT1 = pi
        t1pi_inT1=t1pi

    pi_SyN_transform1 = ants.registration(nmt, t1pi_inT1, type_of_transform='SyNAggro',
                                          reg_iterations=(80, 40, 20), flow_sigma=10, aff_sampling=32,syn_sampling=64,
                                          multivariate_extras=(
                                              ('mattes', nmt, t1pi_inT1, 0.4, 32),
                                              ('MeanSquares', nmt, t1pi_inT1, 0.6, 32)),
                                          outprefix=SUBJECT_DIR + '/PI_T1/xfms/PIinT1_to_NMT_SyN_')

    # t1pi_inNMT = ants.apply_transforms(nmt, t1pi_inT1, pi_SyN_transform1['fwdtransforms'], 'bSpline')
    # pi_inNMT = ants.apply_transforms(nmt, pi_inT1, pi_SyN_transform1['fwdtransforms'], 'bSpline')
    # ants.image_write(pi_inNMT, SUBJECT_DIR + '/PI_T1/atlas/PI_inNMT_0.25mm.nii.gz')
    # ants.image_write(t1pi_inNMT, SUBJECT_DIR + '/PI_T1/atlas/T1PI_inNMT_0.25mm.nii.gz')

    # atlas_inT1 = ants.apply_transforms(t1pi_inT1, atlas, pi_SyN_transform1['invtransforms'], 'multiLabel')
    atlas1_inT1 = ants.apply_transforms(t1, atlas1, pi_SyN_transform1['invtransforms'], 'multiLabel')
    atlas2_inT1 = ants.apply_transforms(t1, atlas2, pi_SyN_transform1['invtransforms'], 'multiLabel')
    # ants.image_write(atlas_inT1, SUBJECT_DIR + '/PI_T1/atlas/D99_in_T1_0.25mm.nii.gz')
    ants.image_write(atlas1_inT1, SUBJECT_DIR + '/PI_T1/atlas/SARM6_in_T1_0.25mm.nii.gz')

    # seg_inT1 = ants.apply_transforms(t1pi_inT1, seg, pi_SyN_transform1['invtransforms'], 'multiLabel')
    # nmt_inT1 = ants.apply_transforms(t1pi_inT1, nmt, pi_SyN_transform1['invtransforms'], 'bSpline')
    if os.path.exists(SUBJECT_DIR + '/PI_T1/T1w_brain_bc_L.nii.gz'):
        # atlas_inPI = ants.apply_transforms(pi, atlas_inT1, pi_SyN_transform0['invtransforms'], 'multiLabel')
        atlas1_inPI = ants.apply_transforms(pi, atlas1_inT1, pi_SyN_transform0['invtransforms'], 'multiLabel')
        atlas2_inPI = ants.apply_transforms(pi, atlas2_inT1, pi_SyN_transform0['invtransforms'], 'multiLabel')
        # seg_inPI = ants.apply_transforms(pi, seg_inT1, pi_SyN_transform0['invtransforms'], 'multiLabel')
        # nmt_inPI = ants.apply_transforms(pi, nmt_inT1, pi_SyN_transform0['invtransforms'], 'bSpline')
    else:
        # atlas_inPI = atlas_inT1
        atlas1_inPI=atlas1_inT1
        atlas2_inPI = atlas2_inT1
        # seg_inPI = seg_inT1
        # nmt_inPI=nmt_inT1
    # ants.image_write(atlas_inPI, SUBJECT_DIR + '/PI_T1/atlas/D99_in_PI_0.25mm.nii.gz')
    ants.image_write(atlas1_inPI, SUBJECT_DIR + '/PI_T1/atlas/SARM6_in_PI_0.25mm.nii.gz')
    ants.image_write(atlas2_inPI, SUBJECT_DIR + '/PI_T1/atlas/subcortex.nii.gz')
    # ants.image_write(seg_inPI, SUBJECT_DIR + '/PI_T1/atlas/seg_in_PI_0.25mm.nii.gz')
    # ants.image_write(nmt_inPI, SUBJECT_DIR + '/PI_T1/atlas/NMT_in_PI_0.25mm.nii.gz')



def atlas_to_t1pi2():
    LOGGER.info('atlas_to_t1pi')
    # NMT is as a bridge
    if not os.path.exists(SUBJECT_DIR + '/PI_T1/atlas'):
        os.makedirs(SUBJECT_DIR + '/PI_T1/atlas')
    nmt = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/NMT_v2.0_asym_brain_L.nii.gz')
    t1pi = ants.image_read(SUBJECT_DIR + '/PI_T1/T1PI_0.25mm.nii.gz')
    pi = ants.image_read(SUBJECT_DIR + '/PI_T1/PI_brain_bc_0.25mm_.nii.gz')
    atlas = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/D99_atlas_in_NMT_v2.0_sym_L_edit.nii.gz')
    seg = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/NMT_v2.0_sym_segmentation_L.nii.gz')
    if os.path.exists(SUBJECT_DIR + '/PI_T1/T1w_brain_bc_L.nii.gz'):
        LOGGER.warning('PI <-T1w-> NMT')
        t1 = ants.image_read(SUBJECT_DIR + '/PI_T1/T1w_brain_bc_L.nii.gz')
        pi_SyN_transform0 = ants.registration(t1, nmt, type_of_transform='SyN',
                                             reg_iterations=(10, 10, 0), flow_sigma=20,grad_step=0.4,syn_sampling=32, multivariate_extras=(
                ('mattes', t1, nmt, 0.2, 32), ('MeanSquares', t1, nmt, 0.8, 32)),
                                             outprefix=SUBJECT_DIR + '/PI_T1/xfms/NMT_to_T1w_SyN_')
        nmt_inT1 = ants.apply_transforms(t1, nmt, pi_SyN_transform0['fwdtransforms'], 'bSpline')
        atlas_inT1 = ants.apply_transforms(t1, atlas, pi_SyN_transform0['fwdtransforms'], 'multiLabel')
        seg_inT1 = ants.apply_transforms(t1, seg, pi_SyN_transform0['fwdtransforms'], 'multiLabel')
        ants.image_write(nmt_inT1, SUBJECT_DIR + '/PI_T1/atlas/nmt_inT1_0.25mm.nii.gz')

    else:
        LOGGER.warning('No T1 image; PI <--> NMT')
        nmt_inT1 = nmt
        atlas_inT1 = atlas
        seg_inT1=seg
    pi_SyN_transform1 = ants.registration(t1pi, nmt_inT1, type_of_transform='SyNAggro',
                                         reg_iterations=(40, 20, 0), flow_sigma=10,syn_sampling=32, multivariate_extras=(
            ('mattes', t1pi, nmt_inT1, 0.4, 32), ('MeanSquares', t1pi, nmt_inT1, 0.6, 32)),
                                         outprefix=SUBJECT_DIR + '/PI_T1/xfms/NMT_to_T1PI_SyN_')

    nmt_inT1PI1 = ants.apply_transforms(t1pi, nmt_inT1, pi_SyN_transform1['fwdtransforms'], 'bSpline')
    atlas_inT1PI1 = ants.apply_transforms(t1pi, atlas_inT1, pi_SyN_transform1['fwdtransforms'], 'multiLabel')
    seg_inT1PI1 = ants.apply_transforms(t1pi, seg_inT1, pi_SyN_transform1['fwdtransforms'], 'multiLabel')

    pi_inT1 = ants.apply_transforms(nmt_inT1, pi, pi_SyN_transform1['invtransforms'], 'bSpline')
    if os.path.exists(SUBJECT_DIR + '/PI_T1/T1w_brain_bc_L.nii.gz'):
        pi_inNMT = ants.apply_transforms(nmt, pi_inT1, pi_SyN_transform0['invtransforms'], 'bSpline')
    else:
        pi_inNMT=pi_inT1

    nmt_inPI=nmt_inT1PI1
    atlas_inPI=atlas_inT1PI1
    ants.image_write(nmt_inPI, SUBJECT_DIR + '/PI_T1/atlas/nmt_inT1PI_0.25mm.nii.gz')
    ants.image_write(atlas_inPI, SUBJECT_DIR + '/PI_T1/atlas/D99_in_PI_0.25mm.nii.gz')
    ants.image_write(seg_inT1PI1,SUBJECT_DIR + '/PI_T1/atlas/seg_in_PI_0.25mm.nii.gz')
    ants.image_write(pi_inNMT,SUBJECT_DIR + '/PI_T1/atlas/PI_in_NMT_0.25mm.nii.gz')

def mymovefile(srcfile,dstpath):                       # 移动函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.move(srcfile, dstpath + fname)          # 移动文件
        print ("move %s -> %s"%(srcfile, dstpath + fname))


def atlas_to_t1pi():
    LOGGER.info('atlas_to_t1pi')
    # T1PI is as a bridge
    if not os.path.exists(SUBJECT_DIR + '/PI_T1/atlas'):
        os.makedirs(SUBJECT_DIR + '/PI_T1/atlas')
    # NMT_v2.0_asym_brain_L.nii.gz
    # nmt = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/D99/NMT_v2.0_asym_brain_L_edit.nii.gz')
    # nmt = ants.image_read('/media/dell/data2_zjLab/macaque/Data_TMP/NMT_D99_ours/NMT_v2.0_sym_SS.nii.gz')
    # nmt = ants.image_read('/media/dell/data2_zjLab/macaque/Data_TMP/NMT_0.5mm/NMT_v2.0_asym_05mm_SS.nii.gz')
    # nmt = ants.image_read('/media/dell/data2_zjLab/macaque/Data_TMP/Macaque/civm/civm_rhesus_v1_b0_oc.nii.gz')
    nmt = ants.image_read('/media/dell/data2_zjLab/macaque/Data_TMP/Macaque/civm/civm_rhesus_v1_dwi_downsample2_oc.nii.gz')
    if not os.path.exists(SUBJECT_DIR + '/PI_T1/PI_brain_bc_icxyz_0.25mm.nii.gz'):
        LOGGER.warning('Loading T1PI_correction_0.25mm.nii.gz')
        t1pi = ants.image_read(SUBJECT_DIR + '/PI_T1/T1PI_correction_0.25mm.nii.gz')
    else:
        LOGGER.warning('Loading PI_brain_bc_icxyz_0.25mm.nii.gz')
        t1pi = ants.image_read(SUBJECT_DIR + '/PI_T1/PI_brain_bc_icxyz_0.25mm.nii.gz')
    pi = ants.image_read(SUBJECT_DIR + '/PI_T1/PI_brain_bc_0.25mm.nii.gz')
    # D99_atlas_in_NMT_v2.0_sym_L_edit.nii.gz
    # atlas = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/D99/D99_atlas_v2.0_L_inmt.nii.gz')
    # atlas = ants.image_read('/media/dell/data2_zjLab/macaque/Data_TMP/NMT_D99_ours/D99_atlas_in_NMT_v2.0_asym.nii.gz')
    # atlas = ants.image_read('/media/dell/data2_zjLab/macaque/Data_TMP/NMT_0.5mm/D99_atlas_in_NMT_v2.0_asym_05mm.nii.gz')
    # atlas = ants.image_read('/media/dell/data2_zjLab/macaque/Data_TMP/Macaque/civm/civm_rhesus_v1_labels_oc.nii.gz')
    atlas = ants.image_read('/media/dell/data2_zjLab/macaque/Data_TMP/Macaque/civm/civm_rhesus_v1_labels_downsample2_oc.nii.gz')
    atlas1 = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/supplemental_SARM/SARM_6_in_NMT_v2.0_sym_edit_L.nii.gz')
    atlas2 = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/cerebellum_L.nii.gz')
    atlas3 = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/supplemental_SARM/subcortex.nii.gz')
    # ventricle=ants.mask_image(nmt,atlas1,level=1)
    # nmt_tmp=nmt -ventricle
    # ventricle_data=ventricle[:,:,:]
    # ventricle_data[ventricle_data>0]=850
    # ventricle[:,:,:]=ventricle_data
    # nmt=nmt_tmp+ventricle

    # atlas1_data=atlas1[:,:,:]
    # atlas1_data[atlas1_data==1]=0
    # atlas1[:, :, :]=atlas1_data
    # ants.image_write(nmt,SUBJECT_DIR + '/PI_T1/tmp/nmt_L.nii.gz')
    # seg = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/NMT_v2.0_sym_segmentation_L.nii.gz')
    if os.path.exists(SUBJECT_DIR + '/PI_T1/T1w_brain_bc_L.nii.gz'):
        LOGGER.warning('PI <-T1w-> NMT')
        t1 = ants.image_read(SUBJECT_DIR + '/PI_T1/T1w_brain_bc_L.nii.gz')
        pi_SyN_transform0 = ants.registration(t1, t1pi, type_of_transform='SyN',reg_iterations=(40, 20, 0), flow_sigma=3,syn_sampling=32,
                                              multivariate_extras=(('mattes', t1, t1pi, 0.6, 0), ('MeanSquares', t1, t1pi, 0.4, 0)),
                                              outprefix=SUBJECT_DIR + '/PI_T1/xfms/PI_to_T1w_SyN_')
        t1pi_inT1 = ants.apply_transforms(t1, t1pi, pi_SyN_transform0['fwdtransforms'], 'bSpline')
        pi_inT1 = ants.apply_transforms(t1, pi, pi_SyN_transform0['fwdtransforms'], 'bSpline')

        T1_inpi = ants.apply_transforms(pi,t1, pi_SyN_transform0['invtransforms'], 'bSpline')
        ants.image_write(T1_inpi, SUBJECT_DIR + '/PI_T1/atlas/T1_inPI_0.25mm.nii.gz')
        ants.image_write(pi_inT1, SUBJECT_DIR + '/PI_T1/atlas/PI_inT1_0.25mm.nii.gz')
        ants.image_write(t1pi_inT1, SUBJECT_DIR + '/PI_T1/atlas/T1PI_inT1_0.25mm.nii.gz')

    else:
        LOGGER.warning('No T1 image; PI <--> NMT')
        # pi_inT1 = pi
        # t1pi_inT1=t1pi
        pi_SyN_transform0 = ants.registration(nmt, t1pi, type_of_transform='SyN',reg_iterations=(40, 20, 0), flow_sigma=3,syn_sampling=32,
                                              multivariate_extras=(('mattes', nmt, t1pi, 0.2, 0), ('MeanSquares', nmt, t1pi, 0.2, 0),('demons', nmt, t1pi, 0.6, 0)),
                                              outprefix=SUBJECT_DIR + '/PI_T1/xfms/PI_to_T1w_SyN_')
        t1pi_inT1 = ants.apply_transforms(nmt, t1pi, pi_SyN_transform0['fwdtransforms'], 'bSpline')
        pi_inT1 = ants.apply_transforms(nmt, pi, pi_SyN_transform0['fwdtransforms'], 'bSpline')


    pi_SyN_transform1 = ants.registration(nmt, t1pi_inT1, type_of_transform='SyNAggro', #SyNAggro
                                          reg_iterations=(80, 60, 10), flow_sigma=2,syn_sampling=32,
                                          multivariate_extras=(
                                              ('mattes', nmt, t1pi_inT1, 0.3, 1),
                                              ('MeanSquares', nmt, t1pi_inT1, 0.4, 1),
                                              ('demons', nmt, t1pi, 0.3, 1)),
                                          outprefix=SUBJECT_DIR + '/PI_T1/xfms/PIinT1_to_NMT_SyN_')

    t1pi_inNMT = ants.apply_transforms(nmt, t1pi_inT1, pi_SyN_transform1['fwdtransforms'], 'bSpline')
    pi_inNMT = ants.apply_transforms(nmt, pi_inT1, pi_SyN_transform1['fwdtransforms'], 'bSpline')
    ants.image_write(pi_inNMT, SUBJECT_DIR + '/PI_T1/atlas/PI_inNMT_0.25mm.nii.gz')
    ants.image_write(t1pi_inNMT, SUBJECT_DIR + '/PI_T1/atlas/T1PI_inNMT_0.25mm.nii.gz')

    atlas_inT1 = ants.apply_transforms(t1pi_inT1, atlas, pi_SyN_transform1['invtransforms'], 'multiLabel')
    atlas1_inT1 = ants.apply_transforms(t1pi_inT1, atlas1, pi_SyN_transform1['invtransforms'], 'multiLabel')
    atlas2_inT1 = ants.apply_transforms(t1pi_inT1, atlas2, pi_SyN_transform1['invtransforms'], 'multiLabel')
    atlas3_inT1 = ants.apply_transforms(t1pi_inT1, atlas3, pi_SyN_transform1['invtransforms'], 'multiLabel')


    ants.image_write(atlas_inT1, SUBJECT_DIR + '/PI_T1/atlas/D99_in_T1_0.25mm.nii.gz')
    ants.image_write(atlas1_inT1, SUBJECT_DIR + '/PI_T1/atlas/SARM6_in_T1_0.25mm.nii.gz')
    ants.image_write(atlas2_inT1, SUBJECT_DIR + '/PI_T1/atlas/cerebellum_mask_in_T1_0.25mm.nii.gz')
    ants.image_write(atlas3_inT1, SUBJECT_DIR + '/PI_T1/atlas/subcortex_inT1_0.25mm.nii.gz')
    # seg_inT1 = ants.apply_transforms(t1pi_inT1, seg, pi_SyN_transform1['invtransforms'], 'multiLabel')
    nmt_inT1 = ants.apply_transforms(t1pi_inT1, nmt, pi_SyN_transform1['invtransforms'], 'bSpline')
    if os.path.exists(SUBJECT_DIR + '/PI_T1/T1w_brain_bc_L.nii.gz'):
        atlas_inPI = ants.apply_transforms(t1pi, atlas_inT1, pi_SyN_transform0['invtransforms'], 'multiLabel')
        atlas1_inPI = ants.apply_transforms(t1pi, atlas1_inT1, pi_SyN_transform0['invtransforms'], 'multiLabel')
        atlas2_inPI = ants.apply_transforms(t1pi, atlas2_inT1, pi_SyN_transform0['invtransforms'], 'multiLabel')
        atlas3_inPI = ants.apply_transforms(t1pi, atlas3_inT1, pi_SyN_transform0['invtransforms'], 'multiLabel')
        # seg_inPI = ants.apply_transforms(t1pi, seg_inT1, pi_SyN_transform0['invtransforms'], 'multiLabel')
        nmt_inPI = ants.apply_transforms(t1pi, nmt_inT1, pi_SyN_transform0['invtransforms'], 'bSpline')
    else:
        # atlas_inPI = atlas_inT1
        # atlas1_inPI=atlas1_inT1
        # atlas2_inPI = atlas2_inT1
        # atlas3_inPI = atlas3_inT1
        # # seg_inPI = seg_inT1
        # nmt_inPI=nmt_inT1
        atlas_inPI = ants.apply_transforms(t1pi, atlas_inT1, pi_SyN_transform0['invtransforms'], 'multiLabel')
        atlas1_inPI = ants.apply_transforms(t1pi, atlas1_inT1, pi_SyN_transform0['invtransforms'], 'multiLabel')
        atlas2_inPI = ants.apply_transforms(t1pi, atlas2_inT1, pi_SyN_transform0['invtransforms'], 'multiLabel')
        atlas3_inPI = ants.apply_transforms(t1pi, atlas3_inT1, pi_SyN_transform0['invtransforms'], 'multiLabel')
        # seg_inPI = ants.apply_transforms(t1pi, seg_inT1, pi_SyN_transform0['invtransforms'], 'multiLabel')
        nmt_inPI = ants.apply_transforms(t1pi, nmt_inT1, pi_SyN_transform0['invtransforms'], 'bSpline')
    ants.image_write(atlas_inPI, SUBJECT_DIR + '/PI_T1/atlas/D99_in_PI_0.25mm.nii.gz')
    ants.image_write(atlas1_inPI, SUBJECT_DIR + '/PI_T1/atlas/SARM6_in_PI_0.25mm.nii.gz')
    ants.image_write(atlas2_inPI, SUBJECT_DIR + '/PI_T1/atlas/cerebellum_mask_in_PI_0.25mm.nii.gz')
    ants.image_write(atlas3_inPI, SUBJECT_DIR + '/PI_T1/atlas/subcortex_inPI_0.25mm.nii.gz')
    # ants.image_write(seg_inPI, SUBJECT_DIR + '/PI_T1/atlas/seg_in_PI_0.25mm.nii.gz')
    ants.image_write(nmt_inPI, SUBJECT_DIR + '/PI_T1/atlas/NMT_in_PI_0.25mm.nii.gz')

def nonlinear_foratlas(f,m,outprefix,s=3,t='SyN'):
    SyN_transform1 = ants.registration(f, m, type_of_transform=t, #SyNAggro
                                          reg_iterations=(40, 20, 0), flow_sigma=s,syn_sampling=32,
                                          outprefix=SUBJECT_DIR + '/PI_T1/xfms/'+outprefix)
    return SyN_transform1

def atlas_to_t1pi3():
    LOGGER.info('atlas_to_t1pi3')
    # T1PI is as a bridge
    if not os.path.exists(SUBJECT_DIR + '/PI_T1/atlas'):
        os.makedirs(SUBJECT_DIR + '/PI_T1/atlas')
    # NMT_v2.0_asym_brain_L.nii.gz
    nmt = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/D99/NMT_v2.0_asym_brain_L_edit.nii.gz')
    if not os.path.exists(SUBJECT_DIR + '/PI_T1/T1PI_correction_xyz_0.25mm.nii.gz'):
        LOGGER.warning('Loading T1PI_correction_0.25mm.nii.gz')
        t1pi = ants.image_read(SUBJECT_DIR + '/PI_T1/T1PI_correction_0.25mm.nii.gz')
    else:
        LOGGER.warning('Loading PI_brain_bc_icxyz_0.25mm.nii.gz')
        t1pi = ants.image_read(SUBJECT_DIR + '/PI_T1/T1PI_correction_xyz_0.25mm.nii.gz')
    pi = ants.image_read(SUBJECT_DIR + '/PI_T1/PI_brain_bc_0.25mm.nii.gz')
    # D99_atlas_in_NMT_v2.0_sym_L_edit.nii.gz
    atlas = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/D99/D99_atlas_v2.0_L_inmt.nii.gz')
    atlas1 = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/supplemental_SARM/SARM_6_in_NMT_v2.0_sym_edit_L.nii.gz')
    atlas2 = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/cerebellum_L.nii.gz')
    atlas3 = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/supplemental_SARM/subcortex.nii.gz')
    ventricle=ants.mask_image(nmt,atlas1,level=1)
    nmt_tmp=nmt -ventricle
    ventricle_data=ventricle[:,:,:]
    ventricle_data[ventricle_data>0]=850
    ventricle[:,:,:]=ventricle_data
    nmt=nmt_tmp+ventricle
    # atlas1_data=atlas1[:,:,:]
    # atlas1_data[atlas1_data==1]=0
    # atlas1[:, :, :]=atlas1_data
    # ants.image_write(nmt,SUBJECT_DIR + '/PI_T1/tmp/nmt_L.nii.gz')
    # seg = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/NMT_v2.0_sym_segmentation_L.nii.gz')
    if os.path.exists(SUBJECT_DIR + '/PI_T1/T1w_brain_bc_L.nii.gz'):
        LOGGER.warning('PI <-T1w-> NMT3')
        t1 = ants.image_read(SUBJECT_DIR + '/PI_T1/T1w_brain_bc_L.nii.gz')

        nmt_tot1=nonlinear_foratlas(t1,nmt,'nmt_tot1')

        nmt_inT1= ants.apply_transforms(t1,nmt, nmt_tot1['fwdtransforms'], 'bSpline')
        atlas_in_NMTinT1 = ants.apply_transforms(t1, atlas, nmt_tot1['fwdtransforms'], 'genericLabel')
        ants.image_write(nmt_inT1, SUBJECT_DIR + '/PI_T1/atlas/nmt_int1_0.25mm.nii.gz')

        t1pi_tot1 = nonlinear_foratlas(t1, t1pi, 't1pi_tot1')
        t1pi_inT1 = ants.apply_transforms(t1, t1pi, t1pi_tot1['fwdtransforms'], 'bSpline')
        ants.image_write(t1pi_inT1, SUBJECT_DIR + '/PI_T1/atlas/t1pi_inT1_0.25mm.nii.gz')

        nmtint1_totipiint1 = nonlinear_foratlas(t1pi_inT1, nmt_inT1, 'nmtint1_totipiint1',s=1,t='SyN')

        nmt_in_tipiint1 = ants.apply_transforms(t1pi_inT1, nmt_inT1, nmtint1_totipiint1['fwdtransforms'], 'bSpline')
        atlas_in_tipiint1 = ants.apply_transforms(t1pi_inT1, atlas_in_NMTinT1, nmtint1_totipiint1['fwdtransforms'], 'genericLabel')

        atlas_int1pi = ants.apply_transforms(t1pi, atlas_in_tipiint1, t1pi_tot1['invtransforms'],'genericLabel')

        nmt_int1pi = ants.apply_transforms(t1pi, nmt_in_tipiint1, t1pi_tot1['invtransforms'], 'bSpline')

        ants.image_write(atlas_int1pi, SUBJECT_DIR + '/PI_T1/atlas/atlas_int1pi_0.25mm_test.nii.gz')
        ants.image_write(nmt_int1pi, SUBJECT_DIR + '/PI_T1/atlas/nmt_int1pi_0.25mm.nii.gz')
    else:
        LOGGER.warning('No T1 image; PI <--> NMT')
        # pi_inT1 = pi
        # t1pi_inT1=t1pi
        pi_SyN_transform0 = ants.registration(nmt, t1pi, type_of_transform='SyN',reg_iterations=(40, 20, 0), flow_sigma=3,syn_sampling=32,
                                              multivariate_extras=(('mattes', nmt, t1pi, 0.2, 0), ('MeanSquares', nmt, t1pi, 0.2, 0),('demons', nmt, t1pi, 0.6, 0)),
                                              outprefix=SUBJECT_DIR + '/PI_T1/xfms/PI_to_T1w_SyN_')
        t1pi_inT1 = ants.apply_transforms(nmt, t1pi, pi_SyN_transform0['fwdtransforms'], 'bSpline')
        pi_inT1 = ants.apply_transforms(nmt, pi, pi_SyN_transform0['fwdtransforms'], 'bSpline')




def atlas_to_t1pi_incomplete():
    LOGGER.info('atlas_to_t1pi')
    # T1PI is as a bridge
    if not os.path.exists(SUBJECT_DIR + '/PI_T1/atlas'):
        os.makedirs(SUBJECT_DIR + '/PI_T1/atlas')

    nmt = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/D99/NMT_v2.0_asym_brain_L_edit.nii.gz')
    t1pi = ants.image_read(SUBJECT_DIR + '/PI_T1/T1PI_correction_0.25mm.nii.gz')

    pi = ants.image_read(SUBJECT_DIR + '/PI_T1/PI_brain_bc_0.25mm.nii.gz')
    pi_mask=ants.image_read(SUBJECT_DIR + '/PI_T1/PI_mask_0.25mm.nii.gz')
    t1pi=pi
    t1pi=ants.mask_image(t1pi,pi_mask)
    atlas = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/D99/D99_atlas_v2.0_L_inmt.nii.gz')
    atlas1 = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/supplemental_SARM/SARM_6_in_NMT_v2.0_sym_edit_L.nii.gz')
    ventricle=ants.mask_image(nmt,atlas1,level=1)
    nmt_tmp=nmt -ventricle
    ventricle_data=ventricle[:,:,:]
    ventricle_data[ventricle_data>0]=850
    ventricle[:,:,:]=ventricle_data
    nmt=nmt_tmp+ventricle

    if os.path.exists(SUBJECT_DIR + '/PI_T1/T1w_brain_bc_L.nii.gz'):
        LOGGER.warning('PI <-T1w-> NMT')
        t1 = ants.image_read(SUBJECT_DIR + '/PI_T1/T1w_brain_bc_L.nii.gz')
        pi_SyN_transform0 = ants.registration(t1, t1pi, type_of_transform='SyN',reg_iterations=(10, 5, 0), flow_sigma=3,syn_sampling=32,
                                              multivariate_extras=(('mattes', t1, t1pi, 0.6, 1), ('MeanSquares', t1, t1pi, 0.4, 1)))


        pimask_inT1 = ants.apply_transforms(t1, pi_mask, pi_SyN_transform0['fwdtransforms'], 'multiLabel')
        t1_=ants.mask_image(t1,pimask_inT1)

        pi_SyN_transform0 = ants.registration(t1_, t1pi, type_of_transform='SyNAggro',reg_iterations=(80, 60, 10), flow_sigma=2,syn_sampling=32,
                                              multivariate_extras=(('mattes', t1, t1pi, 0.6, 1), ('MeanSquares', t1, t1pi, 0.4, 1)),
                                              outprefix=SUBJECT_DIR + '/PI_T1/xfms/PI_to_T1w_SyN_')
        t1pi_inT1 = ants.apply_transforms(t1, t1pi, pi_SyN_transform0['fwdtransforms'], 'bSpline')
        pi_inT1 = ants.apply_transforms(t1, pi, pi_SyN_transform0['fwdtransforms'], 'bSpline')
        ants.image_write(pi_inT1, SUBJECT_DIR + '/PI_T1/atlas/PI_inT1_0.25mm.nii.gz')
        ants.image_write(t1pi_inT1, SUBJECT_DIR + '/PI_T1/atlas/T1PI_inT1_0.25mm.nii.gz')
        ants.image_write(pimask_inT1, SUBJECT_DIR + '/PI_T1/atlas/PImask_inT1_0.25mm.nii.gz')

    else:
        LOGGER.warning('No T1 image; PI <--> NMT')
        pi_inT1 = pi
        t1pi_inT1=t1pi
        pimask_inT1=pi_mask

    pi_SyN_transform1 = ants.registration(nmt, t1pi_inT1, type_of_transform='SyN',
                                          reg_iterations=(10, 5, 0), flow_sigma=3,syn_sampling=32,
                                          multivariate_extras=(
                                              ('mattes', nmt, t1pi_inT1, 0.4, 1),
                                              ('MeanSquares', nmt, t1pi_inT1, 0.6, 1)))

    pimask_inNMT = ants.apply_transforms(nmt, pimask_inT1, pi_SyN_transform1['fwdtransforms'], 'multiLabel')
    nmt_ = ants.mask_image(nmt, pimask_inNMT)

    pi_SyN_transform1 = ants.registration(nmt_, t1pi_inT1, type_of_transform='SyNAggro',
                                          reg_iterations=(80, 60, 10), flow_sigma=2,syn_sampling=32,
                                          multivariate_extras=(
                                              ('mattes', nmt, t1pi_inT1, 0.4, 1),
                                              ('MeanSquares', nmt, t1pi_inT1, 0.6, 1)),
                                          outprefix=SUBJECT_DIR + '/PI_T1/xfms/PIinT1_to_NMT_SyN_')

    t1pi_inNMT = ants.apply_transforms(nmt, t1pi_inT1, pi_SyN_transform1['fwdtransforms'], 'bSpline')
    pi_inNMT = ants.apply_transforms(nmt, pi_inT1, pi_SyN_transform1['fwdtransforms'], 'bSpline')

    ants.image_write(pimask_inNMT, SUBJECT_DIR + '/PI_T1/atlas/PImask_inNMT_0.25mm.nii.gz')
    ants.image_write(pi_inNMT, SUBJECT_DIR + '/PI_T1/atlas/PI_inNMT_0.25mm.nii.gz')
    ants.image_write(t1pi_inNMT, SUBJECT_DIR + '/PI_T1/atlas/T1PI_inNMT_0.25mm.nii.gz')

    atlas_=ants.mask_image(atlas,pimask_inNMT)
    atlas_inT1 = ants.apply_transforms(t1pi_inT1, atlas_, pi_SyN_transform1['invtransforms'], 'multiLabel')
    # atlas1_inT1 = ants.apply_transforms(t1pi_inT1, atlas1, pi_SyN_transform1['invtransforms'], 'multiLabel')

    ants.image_write(atlas_inT1, SUBJECT_DIR + '/PI_T1/atlas/D99_in_T1_0.25mm.nii.gz')
    # ants.image_write(atlas1_inT1, SUBJECT_DIR + '/PI_T1/atlas/SARM6_in_T1_0.25mm.nii.gz')

    # seg_inT1 = ants.apply_transforms(t1pi_inT1, seg, pi_SyN_transform1['invtransforms'], 'multiLabel')
    nmt_inT1 = ants.apply_transforms(t1pi_inT1, nmt_, pi_SyN_transform1['invtransforms'], 'bSpline')
    if os.path.exists(SUBJECT_DIR + '/PI_T1/T1w_brain_bc_L.nii.gz'):
        atlas_inPI = ants.apply_transforms(pi, atlas_inT1, pi_SyN_transform0['invtransforms'], 'multiLabel')
        # atlas1_inPI = ants.apply_transforms(pi, atlas1_inT1, pi_SyN_transform0['invtransforms'], 'multiLabel')
        # seg_inPI = ants.apply_transforms(pi, seg_inT1, pi_SyN_transform0['invtransforms'], 'multiLabel')
        nmt_inPI = ants.apply_transforms(pi, nmt_inT1, pi_SyN_transform0['invtransforms'], 'bSpline')
    else:
        atlas_inPI = atlas_inT1
        # atlas1_inPI=atlas1_inT1
        # seg_inPI = seg_inT1
        nmt_inPI=nmt_inT1
    ants.image_write(atlas_inPI, SUBJECT_DIR + '/PI_T1/atlas/D99_in_PI_0.25mm.nii.gz')
    # ants.image_write(atlas1_inPI, SUBJECT_DIR + '/PI_T1/atlas/SARM6_in_PI_0.25mm.nii.gz')
    # ants.image_write(seg_inPI, SUBJECT_DIR + '/PI_T1/atlas/seg_in_PI_0.25mm.nii.gz')
    ants.image_write(nmt_inPI, SUBJECT_DIR + '/PI_T1/atlas/NMT_in_PI_0.25mm.nii.gz')


def t1pi_to_t1(t1pi):
    imgs = []
    if os.path.exists(SUBJECT_DIR + '/PI_T1/T1w_brain_bc_L.nii.gz'):
        t1 = ants.image_read(SUBJECT_DIR + '/PI_T1/T1w_brain_bc_L.nii.gz')
        # t1pi = ants.image_read(SUBJECT_DIR + '/PI_T1/T1PI_correction_0.25mm.nii.gz')
        pi_SyN_transform = ants.registration(t1, t1pi, type_of_transform='SyN',
                                             reg_iterations=(10, 5, 0), multivariate_extras=(
                ('mattes', t1, t1pi, 0.7, 0), ('MeanSquares', t1, t1pi, 0.3, 0)),
                                             outprefix=SUBJECT_DIR + '/PI_T1/xfms/T1PI_to_T1w_SyN_')
        pi_SyN = ants.apply_transforms(t1, t1pi, pi_SyN_transform['fwdtransforms'], 'bSpline')
        ants.image_write(pi_SyN, SUBJECT_DIR + '/PI_T1/tmp/T1PI_to_T1w_SyN.nii.gz')
        imgs.append(pi_SyN)
        imgs.append(pi_SyN_transform['fwdtransforms'])
        imgs.append(pi_SyN_transform['invtransforms'])
    else:
        LOGGER.warning('No T1 image; PI <--> NMT')
    return imgs


def t1pi_to_t1_to_nmt(img):
    imgs = []
    nmt = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/NMT_v2.0_asym_brain_L_edit.nii.gz')
    pi_SyN = img
    pi_SyN_transform1 = ants.registration(nmt, pi_SyN, type_of_transform='SyNAggro',  # SyNAggro
                                          reg_iterations=(40, 20, 0), multivariate_extras=(
            ('mattes', nmt, pi_SyN, 0.6, 32),
            ('MeanSquares', nmt, pi_SyN, 0.4, 32)), outprefix=SUBJECT_DIR + '/PI_T1/xfms/T1PI_to_T1w_to_NMT_SyN_')
    pi_SyN1 = ants.apply_transforms(nmt, pi_SyN, pi_SyN_transform1['fwdtransforms'], 'bSpline')
    ants.image_write(pi_SyN1, SUBJECT_DIR + '/PI_T1/tmp/T1PI_to_T1w_to_NMT_SyN.nii.gz')
    ####################
    # pi_SyN_transform2 = ants.registration(nmt, pi_SyN1, type_of_transform='SyNAggro',
    #                                       reg_iterations=(100, 100, 100), multivariate_extras=(
    #         ('mattes', nmt, pi_SyN, 0.2, 0),
    #         ('MeanSquares', nmt, pi_SyN, 0.8, 2)), outprefix=SUBJECT_DIR + '/PI_T1/xfms/T1PI_to_T1w_to_NMT_SyN2_')
    # pi_SyN2 = ants.apply_transforms(nmt, pi_SyN1, pi_SyN_transform2['fwdtransforms'], 'bSpline')
    #
    # ants.image_write(pi_SyN2, SUBJECT_DIR + '/PI_T1/tmp/T1PI_to_T1w_to_NMT_SyN.nii.gz')

    imgs.append(pi_SyN1)
    imgs.append(pi_SyN_transform1['fwdtransforms'])
    imgs.append(pi_SyN_transform1['invtransforms'])
    NMT_SyN = ants.apply_transforms(pi_SyN, nmt, pi_SyN_transform1['invtransforms'], 'bSpline')
    imgs.append(NMT_SyN)
    # imgs.append(pi_SyN2)
    # imgs.append(pi_SyN_transform2['fwdtransforms'])
    # imgs.append(pi_SyN_transform2['invtransforms'])
    return imgs


def atlas_correction():
    LOGGER.info('Atlas Correction')
    atlas = ants.image_read(SUBJECT_DIR + '/PI_T1/atlas/D99_in_PI_0.25mm.nii.gz')
    mask = ants.image_read(SUBJECT_DIR + '/PI_T1/PI_mask_0.25.nii.gz')
    atlas_c = ants.mask_image(atlas, mask)
    ants.image_write(atlas_c, SUBJECT_DIR + '/PI_T1/atlas/D99_in_PI_correction_0.25mm.nii.gz')


def up_sample_to_50():
    LOGGER.info('Up-sample to ' + str(SCALE) + 'mm')
    if not os.path.exists(SUBJECT_DIR + '/PI_T1/tmp/PI_rm_dn_icxyz_' + str(SCALE) + 'mm.nii.gz'):
        pi_75 = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rm_dn_' + str(SCALE) + 'mm.nii.gz')
        pi = ants.image_read(SUBJECT_DIR + '/PI_T1/PI_brain_bc_0.25mm.nii.gz')
    else:
        pi_75 = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rm_dn_icxyz_' + str(SCALE) + 'mm.nii.gz')
        pi = ants.image_read(SUBJECT_DIR + '/PI_T1/PI_brain_bc_icxyz_0.25mm.nii.gz')
    if not os.path.exists(SUBJECT_DIR + '/PI_T1/atlas/D99_inPI_contrain_0.25mm.nii.gz'):
        LOGGER.warning('Loading D99_in_PI_0.25mm.nii.gz')
        atlas = ants.image_read(SUBJECT_DIR + '/PI_T1/atlas/D99_in_PI_0.25mm.nii.gz')
    else:
        LOGGER.warning('Loading D99_inPI_contrain_0.25mm.nii.gz')
        atlas = ants.image_read(SUBJECT_DIR + '/PI_T1/atlas/D99_inPI_contrain_0.25mm.nii.gz')

    atlas2 = ants.image_read(SUBJECT_DIR + '/PI_T1/atlas/SARM6_in_PI_0.25mm.nii.gz')
    atlas3 = ants.image_read(SUBJECT_DIR + '/PI_T1/atlas/cerebellum_mask_in_PI_0.25mm.nii.gz')

    pi25_to_pi75_affine_transforms = ants.registration(pi_75, pi, type_of_transform='Affine',    #TRSAA
                                                       reg_iterations=(40, 20, 0), aff_metric='mattes',
                                                       syn_metric='mattes',
                                                       outprefix=SUBJECT_DIR + '/PI_T1/xfms/pi25_to_pi75_affine_')
    pi25_to_pi75 = ants.apply_transforms(pi_75, pi, pi25_to_pi75_affine_transforms['fwdtransforms'], 'bSpline')
    atlas75 = ants.apply_transforms(pi_75, atlas, pi25_to_pi75_affine_transforms['fwdtransforms'], 'multiLabel')
    atlas75_ = ants.apply_transforms(pi_75, atlas2, pi25_to_pi75_affine_transforms['fwdtransforms'], 'multiLabel')
    atlas75_3 = ants.apply_transforms(pi_75, atlas3, pi25_to_pi75_affine_transforms['fwdtransforms'], 'multiLabel')
    ants.image_write(pi25_to_pi75, SUBJECT_DIR + '/PI_T1/tmp/pi25_to_pi_' + str(SCALE) + '.nii.gz')
    if not os.path.exists(SUBJECT_DIR + '/PI_T1/atlas/D99_inPI_contrain_0.25mm.nii.gz'):
        ants.image_write(atlas75, SUBJECT_DIR + '/PI_T1/atlas/D99_in_PI_' + str(SCALE) + 'mm.nii.gz')
    else:
        ants.image_write(atlas75, SUBJECT_DIR + '/PI_T1/atlas/D99_in_PI_contrain_' + str(SCALE) + 'mm.nii.gz')
    ants.image_write(atlas75_, SUBJECT_DIR + '/PI_T1/atlas/SARM6_in_PI_' + str(SCALE) + 'mm.nii.gz')
    ants.image_write(atlas75_3, SUBJECT_DIR + '/PI_T1/atlas/cerebellum_mask_in_PI_' + str(SCALE) + 'mm.nii.gz')


def up_sample_to_0030():
    # LOGGER.info('Up-sample to 0.030mm')
    pi_30 = ants.image_read(SUBJECT_DIR + '/PI_T1/atlas/pi_0.030mm_4.nii.gz')
    pi_75 = ants.image_read(SUBJECT_DIR + '/PI_T1/atlas/pi_0.075mm_4.nii.gz')
    atlas = ants.image_read(SUBJECT_DIR + '/PI_T1/atlas/D99_0.075mm_4.nii.gz')

    atlas30 = te(pi_30, pi_75, atlas)

    ants.image_write(atlas30, SUBJECT_DIR + '/PI_T1/atlas/D99_in_PI_0.030mm_4.nii.gz')


def down_sample_to_0030():
    # LOGGER.info('Down-sample to 0.030mm')
    pi_30 = ants.image_read(SUBJECT_DIR + '/PI_T1/atlas/pi_0.030mm_4.nii.gz')
    pi_75 = ants.image_read(SUBJECT_DIR + '/PI_T1/atlas/pi_0.075mm_4.nii.gz')
    atlas = ants.image_read(SUBJECT_DIR + '/PI_T1/atlas/D99_0.075mm_4.nii.gz')

    atlas30 = te(pi_30, pi_75, atlas)

    ants.image_write(atlas30, SUBJECT_DIR + '/PI_T1/atlas/D99_in_PI_0.030mm_4.nii.gz')


def div4(pi_30):
    output = '/mnt/f4cb1796-fbd8-42b0-8743-1e8ebdb9f1d9/macaque/194787/div/'
    # pi_30_div1=np.zeros(pi_30.shape[0],int(pi_30.shape[1] / 2),int(pi_30.shape[2] / 2))
    # pi_30_div2 = np.zeros(pi_30.shape[0], int(pi_30.shape[1] / 2), int(pi_30.shape[2] / 2))
    # pi_30_div3 = np.zeros(pi_30.shape[0], int(pi_30.shape[1] / 2), int(pi_30.shape[2] / 2))
    # pi_30_div4 = np.zeros(pi_30.shape[0], int(pi_30.shape[1] / 2), int(pi_30.shape[2] / 2))

    pi_30_div1 = pi_30[:, 0:int(pi_30.shape[1] / 2), 0:int(pi_30.shape[2] / 2)]
    pi_30_div2 = pi_30[:, 0:int(pi_30.shape[1] / 2), int(pi_30.shape[2] / 2):pi_30.shape[2]]
    pi_30_div3 = pi_30[:, int(pi_30.shape[1] / 2):pi_30.shape[1], 0:int(pi_30.shape[2] / 2)]
    pi_30_div4 = pi_30[:, int(pi_30.shape[1] / 2):pi_30.shape[1], int(pi_30.shape[2] / 2):pi_30.shape[2]]
    pi30_1 = ants.from_numpy(pi_30_div1)
    pi30_1.set_spacing([0.050, 0.050, 0.050])
    pi30_2 = ants.from_numpy(pi_30_div2)
    pi30_2.set_spacing([0.050, 0.050, 0.050])
    pi30_3 = ants.from_numpy(pi_30_div3)
    pi30_3.set_spacing([0.050, 0.050, 0.050])
    pi30_4 = ants.from_numpy(pi_30_div4)
    pi30_4.set_spacing([0.050, 0.050, 0.050])
    ants.image_write(pi30_1, output + 'D99_50_1.nii.gz')
    ants.image_write(pi30_2, output + 'D99_50_2.nii.gz')
    ants.image_write(pi30_3, output + 'D99_50_3.nii.gz')
    ants.image_write(pi30_4, output + 'D99_50_4.nii.gz')

    # return pi30_1, pi30_2, pi30_3, pi30_4


# div4(ants.image_read('/mnt/f4cb1796-fbd8-42b0-8743-1e8ebdb9f1d9/macaque/194787/div/D99_in_PI_50.nii.gz'))
def te(pi30, pi75, atlas):
    pi75_to_pi30_affine_transforms1 = ants.registration(pi30, pi75, type_of_transform='TRSAA',
                                                        reg_iterations=(40, 20, 0), aff_metric='mattes',
                                                        syn_metric='mattes',
                                                        outprefix=SUBJECT_DIR + '/PI_T1/xfms/pi75_to_pi30_1_affine_')
    # pi25_to_pi75 = ants.apply_transforms(pi30, pi75, pi75_to_pi30_affine_transforms1['fwdtransforms'], 'bSpline')
    atlas30 = ants.apply_transforms(pi30, atlas, pi75_to_pi30_affine_transforms1['fwdtransforms'], 'multiLabel')
    return atlas30


def merge_img():
    atlas_1 = ants.image_read(SUBJECT_DIR + '/PI_T1/atlas/D99_in_PI_0.030mm_1.nii.gz')
    atlas_2 = ants.image_read(SUBJECT_DIR + '/PI_T1/atlas/D99_in_PI_0.030mm_2.nii.gz')
    atlas_3 = ants.image_read(SUBJECT_DIR + '/PI_T1/atlas/D99_in_PI_0.030mm_3.nii.gz')
    atlas_4 = ants.image_read(SUBJECT_DIR + '/PI_T1/atlas/D99_in_PI_0.030mm_4.nii.gz')

    pi_30 = ants.image_read(SUBJECT_DIR + '/CH2_30_30_30_8bita.nii.gz')

    pi_30[:, 0:int(pi_30.shape[1] / 2), 0:int(pi_30.shape[2] / 2)] = atlas_1[:, :, :]
    pi_30[:, 0:int(pi_30.shape[1] / 2), int(pi_30.shape[2] / 2):pi_30.shape[2]] = atlas_2[:, :, :]
    pi_30[:, int(pi_30.shape[1] / 2):pi_30.shape[1], 0:int(pi_30.shape[2] / 2)] = atlas_3[:, :, :]
    pi_30[:, int(pi_30.shape[1] / 2):pi_30.shape[1], int(pi_30.shape[2] / 2):pi_30.shape[2]] = atlas_4[:, :, :]

    ants.image_write(pi_30, SUBJECT_DIR + '/PI_T1/atlas/D99_in_PI_0.030mm.nii.gz')


def compute_conv(fm, kernel):
    [h, w] = fm.shape
    k = 3
    r = int(k / 2)
    padding_fm = np.zeros([h + 2, w + 2])  # , np.int)
    rs = np.zeros([h, w])  # , np.int)
    padding_fm[1:h + 1, 1:w + 1] = fm
    for i in range(1, h + 1):
        for j in range(1, w + 1):
            i0 = i - r
            i1 = i + r + 1
            j0 = j - r
            j1 = j + r + 1
            roi = padding_fm[i0:i1, j0:j1]
            rs[i - 1][j - 1] = np.sum(roi * kernel) / 3
    return rs


# 定义卷积核
def kernel_i():
    weights_data = [
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ]
    weights = np.asarray(weights_data)  # , np.int)
    return weights


def bezier_curve(p0, p1, p2, p3, inserted):
    """
    三阶贝塞尔曲线

    p0, p1, p2, p3 - 点坐标，tuple、list或numpy.ndarray类型
    inserted  - p0和p3之间插值的数量
    """

    assert isinstance(p0, (tuple, list, np.ndarray)), u'点坐标不是期望的元组、列表或numpy数组类型'
    assert isinstance(p0, (tuple, list, np.ndarray)), u'点坐标不是期望的元组、列表或numpy数组类型'
    assert isinstance(p0, (tuple, list, np.ndarray)), u'点坐标不是期望的元组、列表或numpy数组类型'
    assert isinstance(p0, (tuple, list, np.ndarray)), u'点坐标不是期望的元组、列表或numpy数组类型'

    if isinstance(p0, (tuple, list)):
        p0 = np.array(p0)
    if isinstance(p1, (tuple, list)):
        p1 = np.array(p1)
    if isinstance(p2, (tuple, list)):
        p2 = np.array(p2)
    if isinstance(p3, (tuple, list)):
        p3 = np.array(p3)

    points = list()
    for t in np.linspace(0, 1, inserted + 2):
        points.append(p0 * np.power((1 - t), 3) + 3 * p1 * t * np.power((1 - t), 2) + 3 * p2 * (1 - t) * np.power(t,
                                                                                                                  2) + p3 * np.power(
            t, 3))

    return np.vstack(points)


def smoothing_base_bezier(date_x, date_y, k=0.5, inserted=10, closed=False):
    """
    基于三阶贝塞尔曲线的数据平滑算法

    date_x  - x维度数据集，list或numpy.ndarray类型
    date_y  - y维度数据集，list或numpy.ndarray类型
    k   - 调整平滑曲线形状的因子，取值一般在0.2~0.6之间。默认值为0.5
    inserted - 两个原始数据点之间插值的数量。默认值为10
    closed  - 曲线是否封闭，如是，则首尾相连。默认曲线不封闭
    """

    assert isinstance(date_x, (list, np.ndarray)), u'x数据集不是期望的列表或numpy数组类型'
    assert isinstance(date_y, (list, np.ndarray)), u'y数据集不是期望的列表或numpy数组类型'

    if isinstance(date_x, list) and isinstance(date_y, list):
        assert len(date_x) == len(date_y), u'x数据集和y数据集长度不匹配'
        date_x = np.array(date_x)
        date_y = np.array(date_y)
    elif isinstance(date_x, np.ndarray) and isinstance(date_y, np.ndarray):
        assert date_x.shape == date_y.shape, u'x数据集和y数据集长度不匹配'
    else:
        raise Exception(u'x数据集或y数据集类型错误')

    # 第1步：生成原始数据折线中点集
    mid_points = list()
    for i in range(1, date_x.shape[0]):
        mid_points.append({
            'start': (date_x[i - 1], date_y[i - 1]),
            'end': (date_x[i], date_y[i]),
            'mid': ((date_x[i] + date_x[i - 1]) / 2.0, (date_y[i] + date_y[i - 1]) / 2.0)
        })

    if closed:
        mid_points.append({
            'start': (date_x[-1], date_y[-1]),
            'end': (date_x[0], date_y[0]),
            'mid': ((date_x[0] + date_x[-1]) / 2.0, (date_y[0] + date_y[-1]) / 2.0)
        })

    # 第2步：找出中点连线及其分割点
    split_points = list()
    for i in range(len(mid_points)):
        if i < (len(mid_points) - 1):
            j = i + 1
        elif closed:
            j = 0
        else:
            continue

        x00, y00 = mid_points[i]['start']
        x01, y01 = mid_points[i]['end']
        x10, y10 = mid_points[j]['start']
        x11, y11 = mid_points[j]['end']
        d0 = np.sqrt(np.power((x00 - x01), 2) + np.power((y00 - y01), 2))
        d1 = np.sqrt(np.power((x10 - x11), 2) + np.power((y10 - y11), 2))
        k_split = 1.0 * d0 / (d0 + d1)

        mx0, my0 = mid_points[i]['mid']
        mx1, my1 = mid_points[j]['mid']

        split_points.append({
            'start': (mx0, my0),
            'end': (mx1, my1),
            'split': (mx0 + (mx1 - mx0) * k_split, my0 + (my1 - my0) * k_split)
        })

    # 第3步：平移中点连线，调整端点，生成控制点
    crt_points = list()
    for i in range(len(split_points)):
        vx, vy = mid_points[i]['end']  # 当前顶点的坐标
        dx = vx - split_points[i]['split'][0]  # 平移线段x偏移量
        dy = vy - split_points[i]['split'][1]  # 平移线段y偏移量

        sx, sy = split_points[i]['start'][0] + dx, split_points[i]['start'][1] + dy  # 平移后线段起点坐标
        ex, ey = split_points[i]['end'][0] + dx, split_points[i]['end'][1] + dy  # 平移后线段终点坐标

        cp0 = sx + (vx - sx) * k, sy + (vy - sy) * k  # 控制点坐标
        cp1 = ex + (vx - ex) * k, ey + (vy - ey) * k  # 控制点坐标

        if crt_points:
            crt_points[-1].insert(2, cp0)
        else:
            crt_points.append([mid_points[0]['start'], cp0, mid_points[0]['end']])

        if closed:
            if i < (len(mid_points) - 1):
                crt_points.append([mid_points[i + 1]['start'], cp1, mid_points[i + 1]['end']])
            else:
                crt_points[0].insert(1, cp1)
        else:
            if i < (len(mid_points) - 2):
                crt_points.append([mid_points[i + 1]['start'], cp1, mid_points[i + 1]['end']])
            else:
                crt_points.append([mid_points[i + 1]['start'], cp1, mid_points[i + 1]['end'], mid_points[i + 1]['end']])
                crt_points[0].insert(1, mid_points[0]['start'])

    # 第4步：应用贝塞尔曲线方程插值
    out = list()
    group = []
    for item in crt_points:
        inserted = item[3][0] - item[0][0] - 1
        group = bezier_curve(item[0], item[1], item[2], item[3], inserted)
        out.append(group[:-1])

    out.append(group[-1:])
    out = np.vstack(out)

    return out.T[0], out.T[1]


def sagittal():
    remove_edge_light = CONFIG['remove_edge_light']
    rm_bias = CONFIG['remove_edge_light_length']
    rm_value = CONFIG['rm_background_threshold']
    rl = 'L'
    pi_origin = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rm_h.nii.gz')
    # mask = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/cerebellum_mask_' + str(SCALE) + 'mm.nii.gz')
    # pi = ants.mask_image(pi_origin, mask, 0)
    pi_origin_data=pi_origin[:,:,:]
    pi_origin_data = np.nan_to_num(pi_origin_data)
    pi_origin_data[pi_origin_data<=0]=0
    pi_origin[:,:,:]=pi_origin_data
    pi=pi_origin
    pi_avg = np.zeros((pi.shape[0], pi.shape[2]))
    pi_bessel = np.zeros((pi.shape[0], pi.shape[2]))
    for i in range(0, pi.shape[0]):

        pi_slice = pi[i, :, :]

        for k in range(0, pi_slice.shape[1]):
            pi_1d = pi_slice[:, k]
            v5 = np.where(pi_1d > rm_value)
            slice_data = pi_1d[v5]

            if len(slice_data) > 10:
                percent_upper = np.percentile(slice_data, 95)
                slice_data = slice_data[slice_data < percent_upper]
                if len(slice_data) > 10:
                    percent_lower = np.percentile(slice_data, 85)
                    slice_data = slice_data[slice_data > percent_lower]
                else:
                    pi_avg[i, k] = 1
                if len(slice_data) > 10:
                    pi_avg[i, k] = np.average(slice_data)
                else:
                    pi_avg[i, k] = 1
            else:
                pi_avg[i, k] = 1

    down = pi_avg[pi_avg.shape[0] - int(0.1 * pi_avg.shape[0]), :]
    up = pi_avg[int(0.1 * pi_avg.shape[0]), :]
    if len(down[down > 1.1]) > len(up[up > 1.1]):
        rl = 'R'
        LOGGER.info('R correction')
        pi_avg = np.flipud(pi_avg)
        LOGGER.info('R correction end')
    pi_avg_1d = np.zeros((1, pi_avg.shape[1]))
    for k in range(0, pi_avg.shape[1]):
        tmp = pi_avg[:, k]
        if len(tmp[tmp > 10.0]) > 5:
            pi_avg_1d[0, k] = np.mean(tmp[tmp > 10.0])
        else:
            pi_avg_1d[0, k] = 0.0

    # plt.plot(pi_avg_1d, label='pi_avg_1d')
    peaks, _ = scipy.signal.find_peaks(pi_avg_1d[0, :], distance=10, height=50, width=5)
    if len(peaks) > 10:
        center_peaks_y = int(len(peaks) / 2)
        dis = 0
        for p in range(center_peaks_y - 5, center_peaks_y + 5):
            dis = dis + peaks[p] - peaks[p - 1]
        dis = int(dis / 10)
        print('dis y: ' + str(dis))
        peaks_ = peaks_restore(peaks, dis, 0, 0)
        peaks_ = peaks_restore(peaks_, dis, pi_avg_1d.shape[1], 1)
    else:
        peaks_ = peaks
    # peaks_ = np.insert(peaks, 0, 0)
    # peaks_ = np.append(peaks_, pi_avg.shape[1] - 1)
    for i in range(0, pi_avg.shape[0]):
        slice_avg = pi_avg[i, :]

        tmp = pi_avg[:, peaks_[1]]
        t = np.mean(tmp[tmp > 10]) - np.std(tmp[tmp > 10])
        if slice_avg[peaks_[1]] - t < 0:
            for p in range(2, len(peaks_)):
                tmp = pi_avg[:, peaks_[p]]
                t = np.mean(tmp[tmp > 10]) - np.std(tmp[tmp > 10])
                if slice_avg[peaks_[p]] - t < 0:
                    slice_avg[peaks_[p - 1]] = slice_avg[peaks_[p]] - 10
                    break

        tmp = pi_avg[:, peaks_[len(peaks_) - 2]]
        t = np.mean(tmp[tmp > 10]) - np.std(tmp[tmp > 10])
        if slice_avg[peaks_[len(peaks_) - 2]] - t < 0:
            for p in range(3, len(peaks_)):
                tmp = pi_avg[:, peaks_[len(peaks_) - p]]
                t = np.mean(tmp[tmp > 10]) - np.std(tmp[tmp > 10])
                if slice_avg[peaks_[len(peaks_) - p]] - t < 0:
                    slice_avg[peaks_[len(peaks_) - p + 1]] = slice_avg[peaks_[len(peaks_) - p]] - 10
                    break

        y = slice_avg[peaks_]

        x = peaks_
        x_curve, y_curve = smoothing_base_bezier(x, y, k=0.3, closed=False)
        # plt.plot(slice_avg, label='$origin$')
        # plt.legend(loc='best')
        # plt.plot(x, y, 'ro')
        # plt.plot(x_curve, y_curve, label='$k=0.3$')
        pi_bessel[i, :] = y_curve
        print('i: ' + str(i) + '  ' + str(pi_avg.shape[0]))

    pi_ratio = pi_bessel / pi_avg
    pi_ratio[pi_avg < 5] = 1
    pi_ratio[pi_ratio > 5] = 1.0
    # pi_ratio[pi_ratio < 0] = 1

    LOGGER.info('ratio correction')
    if remove_edge_light:
        LOGGER.info('remove_edge_light: ' + str(rm_bias))
        for k in range(0, pi_ratio.shape[1]):
            for i in range(pi_ratio.shape[0] - 1, 0, -1):
                tmp = pi_ratio[i - rm_bias:i, k]
                if len(tmp[tmp > 1.010000]) / rm_bias >= 0.80:
                    pi_ratio[i - rm_bias:pi_ratio.shape[0], k] = pi_ratio[i - rm_bias, k]
                    break
    tifffile.imwrite(SUBJECT_DIR + '/PI_T1/tmp/PI_ratio_s_.tif', pi_ratio)
    LOGGER.info('ratio correction end')

    weights = kernel_i()
    pi_ratio = compute_conv(pi_ratio, weights)
    # pi_ratio = compute_conv(pi_ratio, weights)
    # pi_ratio = compute_conv(pi_ratio, weights)
    pi_ratio[pi_avg < 5] = 1
    pi_ratio[pi_ratio > 5] = 1
    pi_ratio[pi_ratio < 0] = 1
    pi_ratio[np.where((pi_ratio >= 0.0) & (pi_ratio < 1.0))] = 1.0
    if rl == 'R':
        pi_avg = np.flipud(pi_avg)
        pi_ratio = np.flipud(pi_ratio)
    for j in range(0, pi.shape[1]):
        pi[:, j, :] = pi_origin[:, j, :] * pi_ratio[:, :]

    tifffile.imwrite(SUBJECT_DIR + '/PI_T1/tmp/PI_avg_s.tif', pi_avg)
    tifffile.imwrite(SUBJECT_DIR + '/PI_T1/tmp/PI_ratio_s.tif', pi_ratio)
    ants.image_write(pi, SUBJECT_DIR + '/PI_T1/PI_rm_' + str(SCALE) + 'mm.nii.gz')
    # plt.show()


def circle_print(total_time=0):
    list_circle = ["\\", "|", "/", "—"]
    for i in range(total_time * 4):
        time.sleep(0.25)
        print("\r{}".format(list_circle[i % 4]), end="", flush=True)

def peaks_restore(peaks, dis, se, status):
    if status == 0:
        peaks[3] = peaks[4] - dis
        peaks[2] = peaks[3] - dis
        peaks[1] = peaks[2] - dis
        peaks[0] = peaks[1] - dis
        n_peaks = int(peaks[0] / dis)
        for p in range(0, n_peaks):
            peaks = np.insert(peaks, 0, peaks[0] - dis)
        if peaks[0] > 0:
            peaks = np.insert(peaks, 0, 0)
    elif status == 1:
        peaks[len(peaks) - 4] = peaks[len(peaks) - 5] + dis
        peaks[len(peaks) - 3] = peaks[len(peaks) - 4] + dis
        peaks[len(peaks) - 2] = peaks[len(peaks) - 3] + dis
        peaks[len(peaks) - 1] = peaks[len(peaks) - 2] + dis
        n_peaks = int((se - 1 - peaks[len(peaks) - 1]) / dis)
        for p in range(0, n_peaks):
            peaks = np.append(peaks, peaks[len(peaks) - 1] + dis)
        if peaks[len(peaks) - 1] < se - 1:
            peaks = np.append(peaks, se - 1)
    return peaks

def horizontal():
    remove_edge_light = CONFIG['remove_edge_light']
    rm_bias = CONFIG['remove_edge_light_length']
    rm_value = CONFIG['rm_background_threshold']
    cl = CONFIG['callosum_rc_length']
    rl = 'L'
    pi = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rmc_' + str(SCALE) + 'mm' + '.nii.gz')
    pi_origin = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_' + str(SCALE) + 'mm' + '.nii.gz')
    pi_avg = np.zeros((pi.shape[0], pi.shape[2]))
    pi_bessel = np.zeros((pi.shape[0], pi.shape[2]))

    for k in range(0, pi.shape[2]):

        pi_slice = pi[:, :, k]

        for i in range(0, pi_slice.shape[0]):

            pi_1d = pi_slice[i, :]
            v5 = np.where(pi_1d > rm_value)
            slice_data = pi_1d[v5]
            # rm low intensity in background and brain
            if len(slice_data) > 10:
                percent_upper = np.percentile(slice_data, 95)
                slice_data = slice_data[slice_data < percent_upper]
                if len(slice_data) > 10:
                    percent_lower = np.percentile(slice_data, 10)
                    slice_data = slice_data[slice_data > percent_lower]
                if len(slice_data) > 10:
                    pi_avg[i, k] = np.mean(slice_data)
                else:
                    pi_avg[i, k] = 0.0001
            else:
                pi_avg[i, k] = 0.0001
    # identify direction of brain
    down = pi_avg[pi_avg.shape[0] - int(0.1 * pi_avg.shape[0]), :]
    up = pi_avg[int(0.1 * pi_avg.shape[0]), :]
    if len(down[down > 1.1]) > len(up[up > 1.1]):
        rl = 'R'
        LOGGER.info('R correction')
        pi_avg = np.flipud(pi_avg)
        LOGGER.info('R correction end')
    # rm area including a part of cerebellum, as the area intensity is confused
    pi_avg_ = pi_avg[:, 0:pi_avg.shape[1]]
    pi_avg_1d = np.zeros((pi_avg.shape[0], 1))
    for i in range(0, pi_avg.shape[0]):
        tmp = pi_avg_[i, :]
        if len(tmp[tmp > 10.0]) > 0:
            pi_avg_1d[i] = np.mean(tmp[tmp > 10])
        else:
            pi_avg_1d[i] = 0
    # pi_avg_1d[pi_avg_1d.shape[0] - 3:pi_avg_1d.shape[0], 0] = 1.0
    peaks, _ = scipy.signal.find_peaks(pi_avg_1d[:, 0], distance=20, height=25, width=10)  ##50
    # add the first and last point
    peaks_ = np.insert(peaks, 0, 0)
    peaks_ = np.append(peaks_, pi_avg.shape[0] - 1)
    pi_avg_1d=pi_avg_1d[:,0]
    pi_avg_1d[pi_avg_1d.shape[0] - 1] = pi_avg_1d[peaks_[len(peaks_) - 2] - 2]
    pi_avg_1d[0] = pi_avg_1d[peaks_[1] - 2]
    # pi_avg[pi_avg.shape[0] - 1, :] = pi_avg[peaks_[len(peaks_) - 2] - 2, :]
    # pi_avg[peaks_[0], :] = pi_avg[peaks_[1] - 2, :]

    y = pi_avg_1d[peaks_]
    x = peaks_
    x_curve, y_curve = smoothing_base_bezier(x, y, k=0.3, closed=False)
    pi_avg_1d[pi_avg_1d<=1]=1
    pi_ratio_tmp=y_curve/pi_avg_1d
    pi_ratio=np.zeros((pi_avg.shape[0],pi_avg.shape[1]))
    for k in range(0, pi_avg.shape[1]):
        pi_ratio[:,k]=pi_ratio_tmp
    # for k in range(0, pi_avg.shape[1]):
    #     slice_avg = pi_avg[:, k]
    #
    #     # get threshold of peak mean-std
    #     tmp = pi_avg[peaks_[len(peaks_) - 2], :]
    #     t = np.mean(tmp[tmp > 10]) - np.std(tmp[tmp > 10])
    #     # at edge of image, some peaks maybe a low intensity so that putting a neighborhood peak's value
    #     if slice_avg[peaks_[len(peaks_) - 2] + 1] - t < 0:
    #         for p in range(3, len(peaks_)):
    #             tmp = pi_avg[peaks_[len(peaks_) - p], :]
    #             t = np.mean(tmp[tmp > 10]) - np.std(tmp[tmp > 10])
    #             if slice_avg[peaks_[len(peaks_) - p] + 1] - t > 0:
    #                 slice_avg[peaks_[len(peaks_) - p + 1]] = slice_avg[peaks_[len(peaks_) - p] + 1]
    #                 break
    #     else:
    #         slice_avg[slice_avg.shape[0] - 1] = slice_avg[peaks_[len(peaks_) - 2]]  # 15
    #
    #     y = slice_avg[peaks_]
    #     x = peaks_
    #     x_curve, y_curve = smoothing_base_bezier(x, y, k=0.3, closed=False)
    #     # plt.plot(x, y, 'ro')
    #     # plt.plot(slice_avg, label='$origin$')
    #     # plt.plot(x_curve, y_curve, label='$k=0.3$')
    #     # plt.legend(loc='best')
    #     pi_bessel[:, k] = y_curve
    #     print('K: ' + str(k) + '  ' + str(pi_avg.shape[1]))

    # pi_ratio = pi_bessel / pi_avg
    pi_ratio[pi_ratio > 8.0] = 1.0
    # pi_ratio[pi_ratio < 0.01] = 1.0
    LOGGER.info('ratio correction iter0')
    # remove ratio map bottom replaced by normal part nearby bottom
    # if remove_edge_light:
    #     LOGGER.info('remove_edge_light: ' + str(rm_bias))
    #     for i in range(0, pi_ratio.shape[0]):
    #         for k in range(0, pi_ratio.shape[1]):
    #             tmp = pi_ratio[i, k:k + rm_bias]
    #             if len(tmp[tmp > 1.010000]) / rm_bias >= 0.80:
    #                 pi_ratio[i, 0:k + rm_bias] = pi_ratio[i, k + rm_bias]
    #                 break
    #     for i in range(0, pi_ratio.shape[0]):
    #         for k in range(pi_ratio.shape[1] - 1, 0, -1):
    #             tmp = pi_ratio[i, k - rm_bias:k]
    #             if len(tmp[tmp > 1.01]) / rm_bias >= 0.80:
    #                 pi_ratio[i, k - rm_bias:pi_ratio.shape[1]] = pi_ratio[i, k - rm_bias]
    #                 break
    pi_ratio[pi_ratio > 8.0] = 1.0
    pi_ratio[pi_ratio < 0.01] = 1.0
    pi_ratio[np.where((pi_ratio >= 0.0) & (pi_ratio < 1.0))] = 1.0
    # pi_ratio[0:cl,:]=np.flipud(pi_ratio[cl:2*cl,:])
    # pi_ratio[0:cl, :] = pi_ratio[cl:2 * cl, :]
    if rl == 'R':
        pi_avg = np.flipud(pi_avg)
        pi_ratio = np.flipud(pi_ratio)
    pi_ratio[pi_ratio > 8.0] = 1.0
    pi_ratio[pi_ratio < 0.01] = 1.0
    LOGGER.info('ratio correction end')
    for j in range(0, pi.shape[1]):
        pi[:, j, :] = pi_origin[:, j, :] * pi_ratio[:, :]
    tifffile.imwrite(SUBJECT_DIR + '/PI_T1/tmp/PI_avg_h.tif', pi_avg)
    tifffile.imwrite(SUBJECT_DIR + '/PI_T1/tmp/PI_ratio_h.tif', pi_ratio)
    ants.image_write(pi, SUBJECT_DIR + '/PI_T1/tmp/PI_rm_h.nii.gz')
    # plt.show()



def sagittal2():
    remove_edge_light = CONFIG['remove_edge_light']
    rm_bias = CONFIG['remove_edge_light_length']
    rm_value = CONFIG['rm_background_threshold']
    rl = 'L'
    pi_origin = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rm_h.nii.gz')
    mask = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/cerebellum_mask_' + str(SCALE) + 'mm.nii.gz')
    pi = ants.mask_image(pi_origin, mask, 0)

    pi_data=pi[:,:,:]
    pi_m=np.mean(pi_data)
    pi_std=np.std(pi_data)
    pi_data[pi_data>(pi_m+pi_std*2)]=0
    pi[:,:,:]=pi_data

    pi_avg = np.zeros((pi.shape[0], pi.shape[2]))
    pi_bessel = np.zeros((pi.shape[0], pi.shape[2]))
    for i in range(0, pi.shape[0]):

        pi_slice = pi[i, :, :]

        for k in range(0, pi_slice.shape[1]):
            pi_1d = pi_slice[:, k]
            v5 = np.where(pi_1d > rm_value)
            slice_data = pi_1d[v5]

            if len(slice_data) > 10:
                percent_upper = np.percentile(slice_data, 90)
                slice_data = slice_data[slice_data < percent_upper]
                if len(slice_data) > 10:
                    percent_lower = np.percentile(slice_data, 80)
                    slice_data = slice_data[slice_data > percent_lower]
                else:
                    pi_avg[i, k] = 1
                if len(slice_data) > 10:
                    pi_avg[i, k] = np.average(slice_data)
                else:
                    pi_avg[i, k] = 1
            else:
                pi_avg[i, k] = 1

    down = pi_avg[pi_avg.shape[0] - int(0.1 * pi_avg.shape[0]), :]
    up = pi_avg[int(0.1 * pi_avg.shape[0]), :]
    if len(down[down > 1.1]) > len(up[up > 1.1]):
        rl = 'R'
        LOGGER.info('R correction')
        pi_avg = np.flipud(pi_avg)
        LOGGER.info('R correction end')
    pi_avg_1d = np.zeros((1, pi_avg.shape[1]))
    for k in range(0, pi_avg.shape[1]):
        tmp = pi_avg[:, k]
        if len(tmp[tmp > 10.0]) > 5:
            pi_avg_1d[0, k] = np.mean(tmp[tmp > 10.0])
        else:
            pi_avg_1d[0, k] = 0.0

    # plt.plot(pi_avg_1d, label='pi_avg_1d')
    peaks, _ = scipy.signal.find_peaks(pi_avg_1d[0, :], distance=10, height=20, width=5)

    peaks_ = np.insert(peaks, 0, 0)
    peaks_ = np.append(peaks_, pi_avg.shape[1] - 1)
    for i in range(0, pi_avg.shape[0]):
        slice_avg = pi_avg[i, :]

        tmp = pi_avg[:, peaks_[1]]
        t = np.mean(tmp[tmp > 10]) - np.std(tmp[tmp > 10])
        if slice_avg[peaks_[1]] - t < 0:
            for p in range(2, len(peaks_)):
                tmp = pi_avg[:, peaks_[p]]
                t = np.mean(tmp[tmp > 10]) - np.std(tmp[tmp > 10])
                if slice_avg[peaks_[p]] - t < 0:
                    slice_avg[peaks_[p - 1]] = slice_avg[peaks_[p]] - 10
                    break

        tmp = pi_avg[:, peaks_[len(peaks_) - 2]]
        t = np.mean(tmp[tmp > 10]) - np.std(tmp[tmp > 10])
        if slice_avg[peaks_[len(peaks_) - 2]] - t < 0:
            for p in range(3, len(peaks_)):
                tmp = pi_avg[:, peaks_[len(peaks_) - p]]
                t = np.mean(tmp[tmp > 10]) - np.std(tmp[tmp > 10])
                if slice_avg[peaks_[len(peaks_) - p]] - t < 0:
                    slice_avg[peaks_[len(peaks_) - p + 1]] = slice_avg[peaks_[len(peaks_) - p]] - 10
                    break

        if len(peaks_)>0:
            y = slice_avg[peaks_]
            y[y < 0] = 1
            x = peaks_
            x_curve, y_curve = smoothing_base_bezier(x, y, k=0.3, closed=False)
            # plt.plot(slice_avg, label='$origin$')
            # plt.legend(loc='best')
            # plt.plot(x, y, 'ro')
            # plt.plot(x_curve, y_curve, label='$k=0.3$')
            pi_bessel[i, :] = y_curve
        else:
            pi_bessel[i, :] = 1
        print('i: ' + str(i) + '  ' + str(pi_avg.shape[0]))

    pi_ratio = pi_bessel / pi_avg
    pi_ratio[pi_avg < 5] = 1
    pi_ratio[pi_ratio > 5] = 1.0
    # pi_ratio[pi_ratio < 0] = 1

    LOGGER.info('ratio correction')
    if remove_edge_light:
        LOGGER.info('remove_edge_light: ' + str(rm_bias))
        for k in range(0, pi_ratio.shape[1]):
            for i in range(pi_ratio.shape[0] - 1, 0, -1):
                tmp = pi_ratio[i - rm_bias:i, k]
                if len(tmp[tmp > 1.010000]) / rm_bias >= 0.80:
                    pi_ratio[i - rm_bias:pi_ratio.shape[0], k] = pi_ratio[i - rm_bias, k]
                    break
    tifffile.imwrite(SUBJECT_DIR + '/PI_T1/tmp/PI_ratio_s_.tif', pi_ratio)
    LOGGER.info('ratio correction end')

    weights = kernel_i()
    pi_ratio = compute_conv(pi_ratio, weights)
    # pi_ratio = compute_conv(pi_ratio, weights)
    # pi_ratio = compute_conv(pi_ratio, weights)
    pi_ratio[pi_avg < 5] = 1
    pi_ratio[pi_ratio > 5] = 1
    pi_ratio[pi_ratio < 0] = 1
    pi_ratio[np.where((pi_ratio >= 0.0) & (pi_ratio < 1.0))] = 1.0
    if rl == 'R':
        pi_avg = np.flipud(pi_avg)
        pi_ratio = np.flipud(pi_ratio)
    for j in range(0, pi.shape[1]):
        pi[:, j, :] = pi_origin[:, j, :] * pi_ratio[:, :]

    tifffile.imwrite(SUBJECT_DIR + '/PI_T1/tmp/PI_avg_s.tif', pi_avg)
    tifffile.imwrite(SUBJECT_DIR + '/PI_T1/tmp/PI_ratio_s.tif', pi_ratio)
    ants.image_write(pi, SUBJECT_DIR + '/PI_T1/PI_rm_' + str(SCALE) + 'mm.nii.gz')
    # plt.show()


def horizontal2():
    LOGGER.info('GFP correction')
    remove_edge_light = CONFIG['remove_edge_light']
    rm_bias = CONFIG['remove_edge_light_length']
    rm_value = CONFIG['rm_background_threshold']
    cl = CONFIG['callosum_rc_length']
    rl = 'L'
    pi = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rmc_' + str(SCALE) + 'mm' + '.nii.gz')
    pi_data=pi[:,:,:]
    pi_m=np.mean(pi_data)
    pi_std=np.std(pi_data)
    pi_data[pi_data>(pi_m+pi_std*2)]=0
    pi[:,:,:]=pi_data
    pi_origin = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_' + str(SCALE) + 'mm' + '.nii.gz')
    pi_avg = np.zeros((pi.shape[0], pi.shape[2]))
    pi_bessel = np.zeros((pi.shape[0], pi.shape[2]))

    for k in range(0, pi.shape[2]):

        pi_slice = pi[:, :, k]

        for i in range(0, pi_slice.shape[0]):

            pi_1d = pi_slice[i, :]
            v5 = np.where(pi_1d > rm_value)
            slice_data = pi_1d[v5]
            # rm low intensity in background and brain
            if len(slice_data) > 10:
                percent_upper = np.percentile(slice_data, 90)
                slice_data = slice_data[slice_data < percent_upper]
                if len(slice_data) > 10:
                    percent_lower = np.percentile(slice_data, 80)
                    slice_data = slice_data[slice_data > percent_lower]
                if len(slice_data) > 10:
                    pi_avg[i, k] = np.average(slice_data)
                else:
                    pi_avg[i, k] = 0.0001
            else:
                pi_avg[i, k] = 0.0001
    # identify direction of brain
    down = pi_avg[pi_avg.shape[0] - int(0.1 * pi_avg.shape[0]), :]
    up = pi_avg[int(0.1 * pi_avg.shape[0]), :]
    if len(down[down > 1.1]) > len(up[up > 1.1]):
        rl = 'R'
        LOGGER.info('R correction')
        pi_avg = np.flipud(pi_avg)
        LOGGER.info('R correction end')
    # rm area including a part of cerebellum, as the area intensity is confused
    pi_avg_ = pi_avg[:, 100:pi_avg.shape[1]]
    pi_avg_1d = np.zeros((pi_avg.shape[0], 1))
    for i in range(0, pi_avg.shape[0]):
        tmp = pi_avg_[i, :]
        if len(tmp[tmp > 10.0]) > 0:
            pi_avg_1d[i] = np.mean(tmp[tmp > 10])
        else:
            pi_avg_1d[i] = 0
    pi_avg_1d[pi_avg_1d.shape[0] - 3:pi_avg_1d.shape[0], 0] = 1.0
    peaks, _ = scipy.signal.find_peaks(pi_avg_1d[:, 0], distance=20, height=25, width=10)  ##50
    # add the first and last point
    peaks_ = np.insert(peaks, 0, 0)
    peaks_ = np.append(peaks_, pi_avg.shape[0] - 1)
    pi_avg[pi_avg.shape[0] - 1, :] = pi_avg[peaks_[len(peaks_) - 2] - 2, :]
    pi_avg[peaks_[0], :] = pi_avg[peaks_[1] - 2, :]
    for k in range(0, pi_avg.shape[1]):
        slice_avg = pi_avg[:, k]

        # get threshold of peak mean-std
        tmp = pi_avg[peaks_[len(peaks_) - 2], :]
        t = np.mean(tmp[tmp > 10]) - np.std(tmp[tmp > 10])
        # at edge of image, some peaks maybe a low intensity so that putting a neighborhood peak's value
        if slice_avg[peaks_[len(peaks_) - 2] + 1] - t < 0:
            for p in range(3, len(peaks_)):
                tmp = pi_avg[peaks_[len(peaks_) - p], :]
                t = np.mean(tmp[tmp > 10]) - np.std(tmp[tmp > 10])
                if slice_avg[peaks_[len(peaks_) - p] + 1] - t > 0:
                    slice_avg[peaks_[len(peaks_) - p + 1]] = slice_avg[peaks_[len(peaks_) - p] + 1]
                    break
        else:
            slice_avg[slice_avg.shape[0] - 1] = slice_avg[peaks_[len(peaks_) - 2]]  # 15

        y = slice_avg[peaks_]
        x = peaks_
        x_curve, y_curve = smoothing_base_bezier(x, y, k=0.3, closed=False)
        # plt.plot(x, y, 'ro')
        # plt.plot(slice_avg, label='$origin$')
        # plt.plot(x_curve, y_curve, label='$k=0.3$')
        # plt.legend(loc='best')
        pi_bessel[:, k] = y_curve
        print('K: ' + str(k) + '  ' + str(pi_avg.shape[1]))

    pi_ratio = pi_bessel / pi_avg
    pi_ratio[pi_ratio > 8.0] = 1.0
    # pi_ratio[pi_ratio < 0.01] = 1.0
    LOGGER.info('ratio correction iter0')
    # remove ratio map bottom replaced by normal part nearby bottom
    if remove_edge_light:
        LOGGER.info('remove_edge_light: ' + str(rm_bias))
        for i in range(0, pi_ratio.shape[0]):
            for k in range(0, pi_ratio.shape[1]):
                tmp = pi_ratio[i, k:k + rm_bias]
                if len(tmp[tmp > 1.010000]) / rm_bias >= 0.80:
                    pi_ratio[i, 0:k + rm_bias] = pi_ratio[i, k + rm_bias]
                    break
        for i in range(0, pi_ratio.shape[0]):
            for k in range(pi_ratio.shape[1] - 1, 0, -1):
                tmp = pi_ratio[i, k - rm_bias:k]
                if len(tmp[tmp > 1.01]) / rm_bias >= 0.80:
                    pi_ratio[i, k - rm_bias:pi_ratio.shape[1]] = pi_ratio[i, k - rm_bias]
                    break
    pi_ratio[pi_ratio > 8.0] = 1.0
    pi_ratio[pi_ratio < 0.01] = 1.0
    pi_ratio[np.where((pi_ratio >= 0.0) & (pi_ratio < 1.0))] = 1.0
    # pi_ratio[0:cl,:]=np.flipud(pi_ratio[cl:2*cl,:])
    pi_ratio[0:cl, :] = pi_ratio[cl:2 * cl, :]
    if rl == 'R':
        pi_avg = np.flipud(pi_avg)
        pi_ratio = np.flipud(pi_ratio)
    pi_ratio[pi_ratio > 8.0] = 1.0
    pi_ratio[pi_ratio < 0.01] = 1.0
    LOGGER.info('ratio correction end')
    for j in range(0, pi.shape[1]):
        pi[:, j, :] = pi_origin[:, j, :] * pi_ratio[:, :]
    tifffile.imwrite(SUBJECT_DIR + '/PI_T1/tmp/PI_avg_h.tif', pi_avg)
    tifffile.imwrite(SUBJECT_DIR + '/PI_T1/tmp/PI_ratio_h.tif', pi_ratio)
    ants.image_write(pi, SUBJECT_DIR + '/PI_T1/tmp/PI_rm_h.nii.gz')
    # plt.show()

def remove_artifact():
    LOGGER.info('START remove artifact')
    if CONFIG['remove_artifact'] == 'adaptive_rm':
        horizontal()
        sagittal()
    elif CONFIG['remove_artifact'] == 'GFP':
        horizontal2()
        sagittal2()
    elif CONFIG['remove_artifact'] == 'fft':
        fftt()


def bias_correction_5():
    image = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rm_dn_0.5mm.nii.gz')
    fix = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rm_dn_0.25mm.nii.gz')
    fm1 = ants.n4_bias_field_correction(image, convergence={"iters": [50, 50, 50, 50], "tol": 1e-7}, shrink_factor=2,
                                        return_bias_field=True)
    # ants.image_write(t1w_bc, SUBJECT_DIR + '/PI_T1/tmp/biasfield_0.5mm_iter1.nii.gz')
    bc = image / fm1
    fm2 = ants.n4_bias_field_correction(bc, convergence={"iters": [50, 50, 50, 50], "tol": 1e-7}, shrink_factor=2,
                                        return_bias_field=True)
    fm = fm1 * fm2

    # fm=fm1[:,:,:]*fm2[:,:,:]
    # fm=ants.from_numpy(fm[:,:,:])
    # fm.set_spacing(fm1.spacing)
    # fm.set_direction(fm1.direction)
    affine_transform = ants.registration(fix, image, type_of_transform='AffineFast', reg_iterations=(40, 20, 0))
    fm = ants.apply_transforms(fix, fm, affine_transform['fwdtransforms'], 'nearestNeighbor')
    fm[0, :, :] = 1.0
    fm[fm.shape[0] - 1, :, :] = 1.0
    fm[:, 0, :] = 1.0
    fm[:, fm.shape[1] - 1, :] = 1.0
    fm[:, :, 0] = 1.0
    fm[:, :, fm.shape[2] - 1] = 1.0
    ants.image_write(fm, SUBJECT_DIR + '/PI_T1/tmp/biasfield_0.25mm.nii.gz')


def bias_correction_2():
    image = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rm_dn_0.25mm.nii.gz')
    fix = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_' + str(SCALE) + 'mm' + '.nii.gz')
    fm = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/biasfield_0.25mm.nii.gz')
    fm[fm < 0.5] = 1.0
    image_ = image / (fm)
    fm1 = ants.n4_bias_field_correction(image_, convergence={"iters": [50, 50, 50, 50], "tol": 1e-7}, shrink_factor=2,
                                        return_bias_field=True)
    affine_transform = ants.registration(fix, image, type_of_transform='Affine', reg_iterations=(40, 20, 0))
    fm = fm * fm1
    fm = ants.apply_transforms(fix, fm, affine_transform['fwdtransforms'], 'nearestNeighbor')
    fm[0, :, :] = 1.0
    fm[fm.shape[0] - 1, :, :] = 1.0
    fm[:, 0, :] = 1.0
    fm[:, fm.shape[1] - 1, :] = 1.0
    fm[:, :, 0] = 1.0
    fm[:, :, fm.shape[2] - 1] = 1.0
    # fm=fm + 1E-12
    fm[fm < 0.5] = 1.0
    ants.image_write(fm, SUBJECT_DIR + '/PI_T1/tmp/fm.nii.gz')
    fix = fix / fm
    ants.image_write(fix, SUBJECT_DIR + '/PI_T1/tmp/PI_' + str(SCALE) + 'mm' + '.nii.gz')
    # image_bc = image_ / (fm1 + 1E-12)
    # ants.image_write(image_bc, SUBJECT_DIR + '/PI_T1/tmp/PI_rm_dn_bc_0.25mm.nii.gz')


def bias_correction_pi(iter=0):
    if iter == 0:
        LOGGER.info('START bias correction iter 0')
        LOGGER.info('step 0')
        bias_correction_5()
        LOGGER.info('step 1')
        bias_correction_2()
    elif iter ==1:
        LOGGER.info('START bias correction iter 1')
        # SUBJECT_DIR + '/PI_T1/PI_rm_dn_0.25mm.nii.gz'  SUBJECT_DIR + '/PI_T1/tmp/PI_rm_dn_bc_0.25mm.nii.gz'
        image = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_brain_0.25mm.nii.gz')
        mask=ants.get_mask(image)
        image_ = ants.n4_bias_field_correction(image,mask, convergence={"iters": [50, 50, 50, 50], "tol": 1e-7},
                                               shrink_factor=2,
                                               return_bias_field=False,rescale_intensities=True)
        image_ = ants.n4_bias_field_correction(image_,mask, convergence={"iters": [50, 50, 50, 50], "tol": 1e-7},
                                               shrink_factor=2,
                                               return_bias_field=False,rescale_intensities=True)
        ants.image_write(image_, SUBJECT_DIR + '/PI_T1/PI_brain_bc_0.25mm.nii.gz')
        print('bias iter1')



def normalize_to_8bit():
    LOGGER.info('START image to 8bit')
    img = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_' + str(SCALE) + 'mm' + '.nii.gz')
    img_data = img[:, :, :]
    percent_upper = np.percentile(img_data, 95)
    img_ = img_data[img_data < percent_upper]
    if np.max(img_) > 255.0:
        space = np.max(img_) - np.min(img_)
        norm = (img - np.min(img[:, :, :].all())) * 255 / (space + 1E-6)
    else:
        LOGGER.info('max < 255')
        norm = img
    ants.image_write(norm, SUBJECT_DIR + '/PI_T1/tmp/PI_' + str(SCALE) + 'mm' + '.nii.gz')


def log(base, x):
    return np.log(x, out=np.zeros_like(x)) / np.log(base, out=np.zeros_like(x))


def normalize_log():
    normal_log = CONFIG['normal_log']
    img = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rm_dn_' + str(SCALE) + 'mm.nii.gz')
    if normal_log:
        LOGGER.info('START image to norm2')
        n = 60
        tmp = img[:, :, :]
        tmp = img[:, :, :] + 2.0
        matrix = np.where(tmp < n)
        for i in range(0, img.shape[2]):
            tmp_ = tmp[:, :, i]
            tmp[:, :, i] = log(n, tmp_) * n
        tmp[matrix] = img[matrix]
        tmp_morm = ants.from_numpy(tmp)
        tmp_morm.set_spacing(img.spacing)
        tmp_morm.set_direction(img.direction)
        ants.image_write(tmp_morm, SUBJECT_DIR + '/PI_T1/tmp/PI_rm_norm_dn_' + str(SCALE) + 'mm.nii.gz')
    else:
        ants.image_write(img, SUBJECT_DIR + '/PI_T1/tmp/PI_rm_norm_dn_' + str(SCALE) + 'mm.nii.gz')
    # return tmp_morm
    # Parallel(n_jobs=50,require='sharedmem')(delayed(test)(i) for i in range(0, img.shape[0]))
    # ants.image_write(tmp_morm,'/mnt/f4cb1796-fbd8-42b0-8743-1e8ebdb9f1d9/macaque/zzb/8.3/PI_0.075mm_8bit_dn_bc3_norm45.nii.gz')
    # print(0)


def mas_cerebellum():
    LOGGER.info('Remove cerebellum')
    if not os.path.exists(SUBJECT_DIR + '/PI_T1/atlas/'):
        os.mkdir(SUBJECT_DIR + '/PI_T1/atlas/')
    fix = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_' + str(SCALE) + 'mm.nii.gz')
    fix_ = ants.resample_image(fix, (0.25, 0.25, 0.25), interp_type=4)

    move = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/NMT_v2.0_asym_brain_L.nii.gz')
    atlas = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/cerebellum_L.nii.gz')
    pi_affine_transform = ants.registration(fix_, move, type_of_transform='SyN', reg_iterations=(40, 20, 0))
    atlas_ = ants.apply_transforms(fix_, atlas, pi_affine_transform['fwdtransforms'], 'multiLabel')
    # atlas_ = ants.morphology(atlas_affine, operation='dilate', radius=3, mtype='binary', shape='ball')
    ants.image_write(atlas_,SUBJECT_DIR + '/PI_T1/atlas/cerebellum_mask_in_PI_0.25mm.nii.gz')
    pi_affine_transform = ants.registration(fix, fix_, type_of_transform='Affine', reg_iterations=(40, 20, 0))
    atlas_h = ants.apply_transforms(fix, atlas_, pi_affine_transform['fwdtransforms'], 'multiLabel')
    mas = ants.mask_image(fix, atlas_h, 0)
    ants.image_write(mas, SUBJECT_DIR + '/PI_T1/tmp/PI_rmc_' + str(SCALE) + 'mm.nii.gz')
    ants.image_write(atlas_h, SUBJECT_DIR + '/PI_T1/tmp/cerebellum_mask_' + str(SCALE) + 'mm.nii.gz')

def rm_cerebellum_in_finalbrian():
    LOGGER.info('Remove cerebellum in finalbrain')
    if not os.path.exists(SUBJECT_DIR + '/PI_T1/PI_brain_bc_icxyz_0.25mm.nii.gz'):
        img=ants.image_read(SUBJECT_DIR + '/PI_T1/PI_brain_bc_0.25mm.nii.gz')
    else:
        img = ants.image_read(SUBJECT_DIR + '/PI_T1/PI_brain_bc_icxyz_0.25mm.nii.gz')
    mask=ants.image_read(SUBJECT_DIR + '/PI_T1/atlas/cerebellum_mask_in_PI_0.25mm.nii.gz')
    cerebellum=ants.mask_image(img,mask)
    img_noc=img-cerebellum
    # ants.image_write(img_noc,SUBJECT_DIR + '/PI_T1/tmp/PI_brain_bc_nocerebellum_0.25mm.nii.gz')
    # ants.image_write(cerebellum, SUBJECT_DIR + '/PI_T1/tmp/cerebellum_0.25mm.nii.gz')
    mask = ants.get_mask(img_noc)
    LOGGER.info('Bias correction iter0')
    image_ = ants.n4_bias_field_correction(img_noc, mask, convergence={"iters": [50, 50, 50, 50], "tol": 1e-7},
                                           shrink_factor=8,
                                           return_bias_field=False, rescale_intensities=False)
    LOGGER.info('Bias correction iter1')
    image_ = ants.n4_bias_field_correction(image_, mask, convergence={"iters": [50, 50, 50, 50], "tol": 1e-7},
                                           shrink_factor=4,
                                           return_bias_field=False, rescale_intensities=False)
    # LOGGER.info('Bias correction iter2')
    # image_ = ants.n4_bias_field_correction(image_, mask, convergence={"iters": [50, 50, 50, 50], "tol": 1e-7},
    #                                        shrink_factor=2,
    #                                        return_bias_field=False, rescale_intensities=False)
    # LOGGER.info('Bias correction iter3')
    # image_ = ants.n4_bias_field_correction(image_, mask, convergence={"iters": [50, 50, 50, 50], "tol": 1e-7},
    #                                        shrink_factor=2,
    #                                        return_bias_field=False, rescale_intensities=False)
    ants.image_write(image_, SUBJECT_DIR + '/PI_T1/tmp/PI_brain_bc_nocerebellum_0.25mm.nii.gz')


def correct_PIdirection(pi):
    # pi=ants.image_read(SUBJECT_DIR+'/PI_T1/tmp/PI_'+str(SCALE)+'mm.nii.gz')
    affine_metrix1 = np.array([[0.05, 0, 0],
                               [0, -0.05, 0],
                               [0, 0, 0.05]])
    pi.set_direction(affine_metrix1)
    ants.image_write(pi, SUBJECT_DIR + '/PI_T1/tmp/PI_' + str(SCALE) + 'mm.nii.gz')


def check_direction():
    LOGGER.info('Check direction')
    pi = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_' + str(SCALE) + 'mm.nii.gz')
    tmp = pi[pi.shape[0] - int(0.1 * pi.shape[0]), :, :]
    tmp1 = pi[int(0.1 * pi.shape[0]), :, :]
    tmp_percent = len(tmp[tmp > 10]) / (pi.shape[1] * pi.shape[2])
    tmp1_percent = len(tmp1[tmp1 > 10]) / (pi.shape[1] * pi.shape[2])
    if tmp1_percent < tmp_percent:
        LOGGER.info('R to L')
        correct_PIdirection(pi)


def cutoff_callosum():
    t = CONFIG['cutoff_callosum_length']
    LOGGER.info('cutoff_callosum : ' + str(t))
    pi = ants.image_read(SUBJECT_DIR + '/PI_T1/PI_rm_' + str(SCALE) + 'mm.nii.gz')
    pi_ = pi[t:pi.shape[0], :, :]
    pi_ = ants.from_numpy(pi_)
    pi_.set_spacing(pi.spacing)
    pi_.set_direction(pi.direction)
    # pi_.set_origin(pi.set_origin)
    ants.image_write(pi_, SUBJECT_DIR + '/PI_T1/PI_rm_' + str(SCALE) + 'mm.nii.gz')
    print(0)


def labels_correct_by_gm():
    atlas = ants.image_read(os.getcwd() + '/template/NMT/0.25mm/D99_atlas_in_NMT_v2.0_sym_L_edit.nii.gz')
    gm_origin = ants.image_read('/media/dell/data2_zjLab/macaque/zzb/11.6/test/atlas/tmp/gm.nii.gz')

    atlas = ants.mask_image(atlas, gm_origin, level=2)

    atlas_mask = atlas.copy()
    atlas_mask_data = atlas_mask[:, :, :]
    atlas_mask_data = atlas_mask_data.astype(np.uint8)
    atlas_mask_data[atlas_mask_data > 0] = 1
    atlas_mask[:, :, :] = atlas_mask_data

    gm_nan = gm_origin - ants.mask_image(gm_origin, atlas_mask)

    tmp_atlas = atlas[:, :, :]
    tmp_atlas = tmp_atlas.astype(np.uint8)

    gm_nan_data = gm_nan[:, :, :]
    gm_nan_data = gm_nan_data.astype(np.uint8)

    n_tmp = None
    nn_tmp = 0

    for iter in range(0, 10000):

        tmp_atlas_ = np.zeros((atlas.shape[0], atlas.shape[1], atlas.shape[2]))
        tmp_atlas_ = tmp_atlas_.astype(np.uint8)

        X, Y, Z = np.where(gm_nan_data == 2)
        num = len(X)

        if n_tmp == num:
            if nn_tmp == 5:
                break
            nn_tmp = nn_tmp + 1
        n_tmp = num
        for xyz in range(0, len(X)):
            x = X[xyz]
            y = Y[xyz]
            z = Z[xyz]
            print('x : ' + str(x) + " y: " + str(y) + " z : " + str(z) + ' count : ' + str(xyz) + ' num : ' + str(
                num - 1) + ' iter : ' + str(iter))
            if gm_nan_data[x, y, z] == 2:
                if tmp_atlas[x, y, z] == 0:
                    if ((x + 2) < gm_nan_data.shape[0]) & ((x - 2) > 0) & ((y + 2) < gm_nan_data.shape[1]) & (
                            (y - 2) > 0) & ((z + 2) < gm_nan_data.shape[2]) & ((z - 2) > 0):
                        x_up = tmp_atlas[x + 1, y, z]
                        x_upp = tmp_atlas[x + 1 + 1, y, z]
                        x_dn = tmp_atlas[x - 1, y, z]
                        x_dnn = tmp_atlas[x - 1 - 1, y, z]
                        y_up = tmp_atlas[x, y + 1, z]
                        y_upp = tmp_atlas[x, y + 1 + 1, z]
                        y_dn = tmp_atlas[x, y - 1, z]
                        y_dnn = tmp_atlas[x, y - 1 - 1, z]
                        z_up = tmp_atlas[x, y, z + 1]
                        z_upp = tmp_atlas[x, y, z + 1 + 1]
                        z_dn = tmp_atlas[x, y, z - 1]
                        z_dnn = tmp_atlas[x, y, z - 1 - 1]

                        xy_up_rr = tmp_atlas[x + 1, y + 1, z]
                        xy_dn_ll = tmp_atlas[x - 1, y - 1, z]
                        xy_dn_lr = tmp_atlas[x - 1, y + 1, z]
                        xy_dn_rl = tmp_atlas[x + 1, y - 1, z]

                        xz_up_rr = tmp_atlas[x + 1, y, z + 1]
                        xz_dn_ll = tmp_atlas[x - 1, y, z - 1]
                        xz_dn_lr = tmp_atlas[x - 1, y, z + 1]
                        xz_dn_rl = tmp_atlas[x + 1, y, z - 1]

                        yz_up_rr = tmp_atlas[x, y + 1, z + 1]
                        yz_dn_ll = tmp_atlas[x, y - 1, z - 1]
                        yz_dn_lr = tmp_atlas[x, y - 1, z + 1]
                        yz_dn_rl = tmp_atlas[x, y + 1, z - 1]

                        dir = [x_up, x_upp, x_dn, x_dnn, y_up, y_upp, y_dn, y_dnn, z_up, z_upp, z_dn, z_dnn, xy_up_rr,
                               xy_dn_ll, xy_dn_lr, xy_dn_rl, xz_up_rr, xz_dn_ll, xz_dn_lr, xz_dn_rl, yz_up_rr, yz_dn_ll,
                               yz_dn_lr, yz_dn_rl]

                        dir = np.asarray(dir, np.uint8)
                        d, n = np.unique(dir, return_counts=True)
                        n_max = 0
                        d_max = 0
                        for i in range(0, len(d)):
                            if not d[i] == 0:
                                if n_max < n[i]:
                                    n_max = n[i]
                                    d_max = d[i]
                        if not d_max == 0:
                            tmp_atlas_[x, y, z] = int(d_max)
        x_, y_, z_ = np.where(tmp_atlas_ > 0)
        gm_nan_data[x_, y_, z_] = 0

        tmp_atlas = tmp_atlas + tmp_atlas_

        img = ants.from_numpy(tmp_atlas)
        img.set_spacing(atlas.spacing)
        img.set_direction(atlas.direction)
        img.set_origin(atlas.origin)

        ants.image_write(img, SUBJECT_DIR + '/PI_T1/tmp/tmp_atlas_iter' + str(iter) + '.nii.gz')

    print('iter : ' + str(iter))
    img = ants.from_numpy(tmp_atlas)
    img.set_spacing(atlas.spacing)
    img.set_direction(atlas.direction)
    img.set_origin(atlas.origin)
    ants.image_write(img,SUBJECT_DIR + '/PI_T1/atlas/D99_in_PI_correction_0.25mm.nii.gz')

def denoising_t1():
    img=ants.image_read(SUBJECT_DIR+'/T1_NMT/T1w_brain_bc.nii.gz')
    mask = ants.image_read(SUBJECT_DIR + '/T1_NMT/T1_brain_mask.nii.gz')
    img=ants.denoise_image(img,mask)
    img = ants.denoise_image(img, mask,noise_model="Gaussian")
    ants.image_write(img,SUBJECT_DIR+'/T1_NMT/T1w_brain_bc_dn.nii.gz')

def seg_t1pi():
    LOGGER.info('seg t1pi')
    if not os.path.exists(SUBJECT_DIR + '/PI_T1/seg/'):
        os.makedirs(SUBJECT_DIR + '/PI_T1/seg/')
        os.makedirs(SUBJECT_DIR + '/PI_T1/seg/tmp')
    img_tmp=ants.image_read(SUBJECT_DIR+'/PI_T1/T1PI_correction_xyz_0.25mm.nii.gz')
    img = ants.image_read(SUBJECT_DIR+'/PI_T1/PI_brain_bc_icxyz_0.25mm.nii.gz')
    subcortex = ants.image_read(SUBJECT_DIR+'/PI_T1/atlas/subcortex_inPI_0.25mm.nii.gz')

    img_subcortex = ants.mask_image(img, subcortex, level=[1])
    img = img - img_subcortex

    subcortex = ants.mask_image(img_tmp, subcortex, level=[1])
    img_tmp = img_tmp - subcortex

    mask = ants.get_mask(img, cleanup=2)
    img_tmp = ants.mask_image(img_tmp, mask)

    # mask = ants.get_mask(img_tmp, low_thresh=10)

    seg = ants.kmeans_segmentation(img_tmp, 2, kmask=mask, mrf=0.1)
    ants.image_write(seg['segmentation'], SUBJECT_DIR + '/PI_T1/seg/tmp/seg0.nii.gz')
    ants.image_write(seg['probabilityimages'][0], SUBJECT_DIR + '/PI_T1/seg/tmp/p0_iter0.nii.gz')
    ants.image_write(seg['probabilityimages'][1], SUBJECT_DIR + '/PI_T1/seg/tmp/p1_iter0.nii.gz')

    priorseg = ants.prior_based_segmentation(img, seg['probabilityimages'], mask, 0.25)
    ants.image_write(priorseg['segmentation'], SUBJECT_DIR + '/PI_T1/seg/tmp/seg1.nii.gz')
    ants.image_write(priorseg['probabilityimages'][0], SUBJECT_DIR + '/PI_T1/seg/tmp/p1_iter1.nii.gz')
    ants.image_write(priorseg['probabilityimages'][1], SUBJECT_DIR + '/PI_T1/seg/tmp/p2_iter1.nii.gz')
    # seg=priorseg['segmentation']
    # wm = ants.mask_image(seg, seg, level=2)
    # wm = ants.morphology(wm, 'erode', radius=5, shape='ball')
    # wm = ants.get_mask(wm, cleanup=2)
    # img=img-ants.mask_image(img,wm)
    # mask=mask-wm
    # priorseg = ants.prior_based_segmentation(img, priorseg['probabilityimages'], mask, 0.25)
    # ants.image_write(priorseg['segmentation'], SUBJECT_DIR + '/PI_T1/seg/tmp/seg2.nii.gz')
    # ants.image_write(priorseg['probabilityimages'][0], SUBJECT_DIR + '/PI_T1/seg/tmp/p1_iter2.nii.gz')
    # ants.image_write(priorseg['probabilityimages'][1], SUBJECT_DIR + '/PI_T1/seg/tmp/p2_iter2.nii.gz')

    gm = ants.mask_image(priorseg['segmentation'], priorseg['segmentation'], level=[1])
    ants.image_write(gm, SUBJECT_DIR + '/PI_T1/seg/gm_mask.nii.gz')

def normlize_slice(img,mask,direction='x'):
    img_ = img
    # direction = 'y'  # x  z
    mask_data = mask[:, :, :]
    # mask_data[mask_data == 104] = 0
    # mask_data[mask_data > 0] = 1
    mask[:, :, :] = mask_data[:, :, :]
    img = ants.mask_image(img_, mask, level=1)
    if direction == 'x':
        tmp = np.zeros((img.shape[0]))
        n = img.shape[0]
    elif direction == 'y':
        tmp = np.zeros((img.shape[1]))
        n = img.shape[1]
    else:
        tmp = np.zeros((img.shape[2]))
        n = img.shape[2]

    for i in range(0, n):
        print('slice : ' + str(i) + ' ALL: ' + str(n - 1) + ' direction: ' + direction)

        if direction == 'x':
            slice = img[i, :, :]
        elif direction == 'y':
            slice = img[:, i, :]
        else:
            slice = img[:, :, i]

        m = np.mean(slice[slice > 10])
        std = np.std(slice[slice > 10])
        slice[slice >= (m + std * 1)] = 0
        slice[slice <= (m - std * 1)] = 0

        if not np.isnan(np.mean(slice[slice > 0])):
            tmp[i] = np.mean(slice[slice > 0])
        else:
            tmp[i] = 0
        print('slice mean : ' + str(tmp[i]))

    max = np.max(tmp)
    tmp[tmp < 0.1] = max
    ratio = max / tmp

    m = np.percentile(ratio, 95)
    ratio[ratio > m] = m
    ratio[0:100] = ratio[100]
    ratio[ratio.shape[0] - 100:ratio.shape[0]] = ratio[ratio.shape[0] - 100]
    for i in range(0, n):
        if direction == 'x':
            img_[i, :, :] = ratio[i] * img_[i, :, :]
        elif direction == 'y':
            img_[:, i, :] = ratio[i] * img_[:, i, :]
        else:
            img_[:, :, i] = ratio[i] * img_[:, :, i]

    # img_data=img_[:,:,:]
    # space = np.max(img_data) - np.min(img_data)
    # norm = (img_data - np.min(img_data)) * 255 / space
    # img_[:,:,:] = norm.astype(np.uint8)
    return img_
    # ants.image_write(img_, '/media/dell/data2_zjLab/macaque/zzb/11.15/PI/194787/PI_T1/tmp/PI_rm_norm_y.nii.gz')

def normlize_img():
    LOGGER.info('Normlize img')
    img=ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rm_norm_dn_ic_' + str(SCALE) + 'mm.nii.gz')
    mask=ants.image_read(SUBJECT_DIR+'/PI_T1/atlas/D99_in_PI_0.05mm.nii.gz')
    mask=mask-ants.mask_image(mask,mask,[57,64,108,109,110,111,112,113,114,115,116,118,119,120,121,125,222,223,224])
    mask_data=mask[:,:,:]

    mask_data[mask_data>0]=1
    mask[:, :, :]=mask_data
    LOGGER.info('Normlize X')
    img_iter1=normlize_slice(img,mask,'x')
    LOGGER.info('Normlize Z')
    img_iter2 = normlize_slice(img_iter1, mask, 'z')
    LOGGER.info('Normlize Y')
    img_iter3 = normlize_slice(img_iter2, mask, 'y')
    ants.image_write(img_iter3,SUBJECT_DIR + '/PI_T1/tmp/PI_rm_dn_icxyz_' + str(SCALE) + 'mm.nii.gz')

def resample_brain():
    LOGGER.info('START resample pi brain')
    pi = ants.image_read(SUBJECT_DIR + '/PI_T1/tmp/PI_rm_dn_icxyz_' + str(SCALE) + 'mm.nii.gz')
    fix = ants.image_read(SUBJECT_DIR + '/PI_T1/PI_brain_bc_0.25mm.nii.gz')
    affine_transform = ants.registration(fix, pi, type_of_transform='TRSAA', reg_iterations=(40, 20, 0))
    pi_25 = ants.apply_transforms(fix, pi, affine_transform['fwdtransforms'], 'bSpline')

    ants.image_write(pi_25,SUBJECT_DIR + '/PI_T1/PI_brain_bc_icxyz_0.25mm.nii.gz')

def gm_contrain():
    LOGGER.info('GM contrain')
    img3 = ants.image_read(SUBJECT_DIR+'/PI_T1/atlas/NMT_in_PI_0.25mm.nii.gz')
    img2 = ants.image_read(SUBJECT_DIR+'/PI_T1/atlas/D99_in_PI_0.25mm.nii.gz')
    fix = ants.image_read(SUBJECT_DIR+'/PI_T1/seg/gm_mask.nii.gz')
    img = img2.copy()
    img_data=img[:,:,:]
    img_data[img_data>0]=1
    img[:, :, :]=img_data
    pi_affine_transform = ants.registration(fix, img, type_of_transform='SyN', reg_iterations=(15, 10, 0),
                                            flow_sigma=5,syn_sampling=32,outprefix=SUBJECT_DIR+'/PI_T1/xfms/gm_constrain_',
                                            multivariate_extras=(
                                                ('mattes', fix, img, 0.2, 10), ('MeanSquares', fix, img, 0.2, 10),('demons', fix, img, 0.6 ,10)))

    img_2 = ants.apply_transforms(fix, img2, pi_affine_transform['fwdtransforms'], 'genericLabel')
    img_3 = ants.apply_transforms(fix, img3, pi_affine_transform['fwdtransforms'], 'bSpline')

    ants.image_write(img_2,SUBJECT_DIR+'/PI_T1/atlas/D99_inPI_contrain_0.25mm.nii.gz')
    ants.image_write(img_3,SUBJECT_DIR+'/PI_T1/atlas/NMT_inPI_contrain_0.25mm.nii.gz')

def g_contraint(fix,mov):
    Gaus = sitk.GradientMagnitudeRecursiveGaussianImageFilter()
    fixed_image=fix.copy()
    moving_image = mov.copy()
    fix_ = sitk.GetImageFromArray(fix[:, :, :])
    mov_ = sitk.GetImageFromArray(mov[:, :, :])
    fixed_image_g = Gaus.Execute(fix_)
    moving_image_g = Gaus.Execute(mov_)
    fixed_image[:,:,:]=sitk.GetArrayFromImage(fixed_image_g)
    moving_image[:, :, :] = sitk.GetArrayFromImage(moving_image_g)

    # affine_transform = ants.registration(fixed_image, moving_image, type_of_transform='TRSAA')
    # img_g = ants.apply_transforms(fixed_image, moving_image ,affine_transform['fwdtransforms'], 'bSpline')

    syn_transform = ants.registration(fixed_image, moving_image, type_of_transform='SyN',flow_sigma=0.1,reg_iterations=(40, 20, 0))
    # img_g = ants.apply_transforms(fixed_image, img_g ,syn_transform['fwdtransforms'], 'bSpline')
    # return affine_transform,syn_transform
    return  syn_transform