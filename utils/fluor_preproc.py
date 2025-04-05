import os
import ants
import yaml


fluor_YAML_PATH = os.getcwd() + '/config/fluor_sections_config.yaml'
fluor_CONFIG = yaml.safe_load(open(fluor_YAML_PATH, 'r'))

def fluo_toBstyle():
    img1 = ants.image_read(fluor_CONFIG['subject_dir'])
