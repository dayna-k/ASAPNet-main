"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os.path
from data.pix2pix_dataset2 import Pix2pixDataset
from data.image_folder import make_dataset


class SARDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(dataroot='D:/Dataset/KOMPSAT5/')
        # D:/Dataset/KOMPSAT5/TRAINA/A
        parser.set_defaults(preprocess_mode='none')
        load_size = 256 if is_train else 256
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=0)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(no_instance_edge=True)
        parser.set_defaults(no_instance_dist=True)
        parser.set_defaults(lr_instance=False)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'test' if opt.phase == 'test-png' else 'TRAIN-png'
        #phase = 'val' if opt.phase == 'test' else opt.phase

        label_dir = os.path.join(root, '%s/A' % phase)
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        image_dir = os.path.join(root, '%s/B' % phase)
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        instance_paths = []

        return label_paths, image_paths, instance_paths
