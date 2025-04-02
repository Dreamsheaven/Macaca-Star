import os

import yaml
import utils.Logger as loggerz
from utils.brain_procession import crop_brain, denoise, resample_image, init_logger, t1pi_correction, mas_pi, \
    make_pi_mask, atlas_to_t1pi, fftt, atlas_correction, up_sample_to_0030, tif_to_nii, \
    remove_artifact, bias_correction, bias_correction_pi, normalize_to_8bit, normalize_log, mask_affine, mas_cerebellum, \
    correct_PIdirection, check_direction, cutoff_callosum, clahe_image, up_sample_to_50, subcor_atlas_to_t1pi, \
    atlas_to_t1pi2, labels_correct_by_gm, subcor_atlas_to_t1pi2, rm_cerebellum_in_finalbrian, atlas_to_t1pi_incomplete, \
    seg_t1pi, normlize_img, resample_brain, gm_contrain, atlas_to_t1pi3
from utils.CycleGan_3D.test import PI_to_T1_cyclegan, T1_to_PI_cyclegan, PI_to_T1_cyclegan_test

YAML_PATH = os.getcwd() + '/config/config.yaml'
CONFIG = yaml.safe_load(open(YAML_PATH, 'r'))
SUBJECT_DIR = CONFIG['subject_dir']

def PI_T1():
    LOGGER = loggerz.LOGGER
    init_logger()
    LOGGER.info('START PI<-->T1')
    if os.path.exists(SUBJECT_DIR + '/T1w.nii.gz'):
        # crop_brain('L')
        print(0)
    else:
        if not os.path.exists(SUBJECT_DIR + '/PI_T1'):
            os.makedirs(SUBJECT_DIR + '/PI_T1')
        if not os.path.exists(SUBJECT_DIR + '/PI_T1/xfms'):
            os.makedirs(SUBJECT_DIR + '/PI_T1/xfms')
    # tif_to_nii()
    # check_direction()
    # normalize_to_8bit()
    # denoise(iter=0)
    # resample_image(iter=0)
    ## make_pi_mask()
    ## mask_affine()
    # bias_correction_pi(iter=0)
    # mas_cerebellum()
    # remove_artifact()
    if CONFIG['cutoff_callosum']:
        cutoff_callosum()
    # denoise(iter=3)
    # normalize_log()
    # clahe_image()
    # resample_image(iter=1)
    # make_pi_mask()
    # mas_pi()
    # bias_correction_pi(iter=1)
    # rm_cerebellum_in_finalbrian()
    # PI_to_T1_cyclegan()
    # t1pi_correction()
    # subcor_atlas_to_t1pi()
    # atlas_to_t1pi()
    # up_sample_to_50()
    # normlize_img()
    # resample_brain()
    # rm_cerebellum_in_finalbrian()
    # PI_to_T1_cyclegan()
    # t1pi_correction()
    atlas_to_t1pi()
    # seg_t1pi()
    # gm_contrain()
    # up_sample_to_50()