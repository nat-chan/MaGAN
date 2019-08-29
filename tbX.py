"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from models.pix2pix_model import Pix2PixModel
from options.test_options import TestOptions
import argparse
import torch
from torchsummary import summary #type:ignore
import pdb
import sys

from tensorboardX import SummaryWriter

_ = argparse.ArgumentParser.parse_args
argparse.ArgumentParser.parse_args = lambda self,*args,**kwargs: _(self, args=['--name', 'coco_pretrained', '--dataset_mode', 'coco', '--dataroot', './datasets/coco_stuff']) #type:ignore

opt = TestOptions().parse()

model = Pix2PixModel(opt)

model.eval()

data = torch.cuda.FloatTensor(1,184,256,256) #type:ignore

with SummaryWriter(comment='SPADE_netG') as w:
    w.add_graph(model.netG, data, True)

summary(model.netG, input_size=(184,256,256))
