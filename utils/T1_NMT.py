import os

import yaml

from utils.brain_procession import brain_orientation_correction, affine, nonlinear, make_brain_mask, bias_correction, \
    make_brain_atlas, init_logger, denoising_t1

import utils.Logger as Logger

YAML_PATH = os.getcwd() + '/config/config.yaml'
CONFIG = yaml.safe_load(open(YAML_PATH, 'r'))
SUBJECT_DIR = CONFIG['subject_dir']


def T1_NMT():
    LOGGER = Logger.LOGGER
    init_logger()
    if os.path.exists(SUBJECT_DIR + '/T1w.nii.gz'):
        # brain_orientation_correction()
        # #affine()
        # nonlinear()
        make_brain_mask()
        # bias_correction()
        # denoising_t1()
    else:
        LOGGER.warning('No T1 image; Next step to run PI <--> NMT')
    # V1 method; Now don't need t1 atlas;
    # make_brain_atlas()
    LOGGER.info('END T1<-->NMT')
