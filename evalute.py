"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from os import path
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
from tqdm import tqdm

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
with tqdm(dataloader, dynamic_ncols=True) as pbar:
    for i, data_i in enumerate(pbar):
        generated = model(data_i, mode='inference')
        img_path = data_i['path']
        for b in range(generated.shape[0]):
            visuals = OrderedDict([
            ('synth' , generated[b])                           ,
            ('real' , data_i['image'][b])                     ,
            ])
            visualizer.save_images(webpage, visuals, img_path[b:b + 1])
            pbar.set_description(opt.which_epoch + ' ' + path.basename(img_path[b]).split('.')[0])

