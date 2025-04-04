import utils.Logger as loggerz
from utils.blockface_preproc import intensity_c, b_alignMRI


def blockface_preproc():
    # intensity_c()
    b_alignMRI()
    b_to_T1_cyclegan()
