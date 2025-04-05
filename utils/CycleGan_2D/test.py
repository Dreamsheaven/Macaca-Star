import os
from utils.CycleGan_2D.options.test_options import TestOptions
from utils.CycleGan_2D.data import create_dataset
from utils.CycleGan_2D.models import create_model


def fluor_toB_cyclegan():
    opt = TestOptions().parse()
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.checkpoints_dir = os.getcwd() + '/checkpoints'
    opt.name = 'fluor2Blockface'
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1
    opt.no_dropout = True # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results







