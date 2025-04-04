import utils.Logger as loggerz
from utils.CycleGan_3D.test import b_to_T1_cyclegan
from utils.blockface_preproc import intensity_c, b_alignMRI, correct_t1like, blockface_3Dreg, b_invetalignMRI, \
    repair_blockface


def blockface_preproc():
    intensity_c()
    b_alignMRI()
    b_to_T1_cyclegan()
    correct_t1like()
    blockface_3Dreg()
    b_invetalignMRI()
    repair_blockface()