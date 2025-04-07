from utils.CycleGan_2D.test import fluor_toB_cyclegan
from utils.fluor_preproc import repaire_blikefluo, fluor_SyNtoB_bySeg


def fluor_preproc():
    fluor_toB_cyclegan()
    repaire_blikefluo()
    fluor_SyNtoB_bySeg()