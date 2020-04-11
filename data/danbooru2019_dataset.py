"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os.path
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import os
import torch


class Danbooru2019Dataset(Pix2pixDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.add_argument('--coco_no_portraits', action='store_true')
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=0)
        return parser

    def initialize(self, opt):
        self.opt = opt
        image_paths, hed_paths = self.get_paths(opt)
        self.hed_paths = hed_paths
        self.image_paths = image_paths
        self.instance_paths = None
        self.label_paths = None
        self.dataset_size = len(self.image_paths)
        self.dummy = torch.Tensor([]).reshape((0,256,256)).cuda()

    def get_paths(self, opt):
        root = opt.dataroot
        with open(os.path.join(root, opt.phase+"_img.txt"), 'r') as f:
            image_paths = f.read().splitlines()
        with open(os.path.join(root, opt.phase+"_hed.txt"), 'r') as f:
            hed_paths = f.read().splitlines()
        return image_paths, hed_paths

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = image.convert('RGB')

        params = get_params(self.opt, image.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        transform_image = get_transform(self.opt, params)

        image_tensor = transform_image(image)

        hed_path = self.hed_paths[index]
        hed = Image.open(hed_path)
        hed = hed.convert('L')
        hed_tensor = transform_label(hed)
        hed_tensor = 1 - hed_tensor # nega -> posi

        input_dict = {'label': self.dummy,
                      'instance': self.dummy,
                      'image': image_tensor.cuda(),
                      'hed': hed_tensor.cuda(),
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

