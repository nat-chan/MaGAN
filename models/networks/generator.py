"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import configargparse as argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock
from models.networks.architecture import SPADEResnetBlock
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d

class ToRGB(nn.Module):
    def __init__(self, in_channels, out_channels=3):
        super().__init__()
        self.conv_img = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    def forward(self, input):
        out = F.leaky_relu(input, 2e-1)
        out = self.conv_img(out)
        out = F.tanh(out)
        return out

class JPUMaGANGenerator(BaseNetwork):
    """
    すべてのlatentをJPUに入力する．pseudo segmentは活性化させる
    ボトルネックをもう一段低解像度にしてreceptive fieldの拡大を試みる
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf #64

        # sharing modules in each hierarchy, number of learning parameters == 0
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.up = nn.Upsample(scale_factor=2, mode=opt.resize_mode, align_corners=opt.resize_align_corners)

        # down
        self.conv_0 = nn.Conv2d(self.opt.semantic_nc , 1  * nf , kernel_size=3 , stride=1   , padding=1)
        self.conv_1 = nn.Conv2d(1  * nf              , 2  * nf , kernel_size=3 , stride=2   , padding=1)
        self.conv_2 = nn.Conv2d(2  * nf              , 4  * nf , kernel_size=3 , stride=2   , padding=1)
        self.conv_3 = nn.Conv2d(4  * nf              , 8  * nf , kernel_size=3 , stride=2   , padding=1)
        self.conv_4 = nn.Conv2d(8  * nf              , 16 * nf , kernel_size=3 , stride=2   , padding=1)
        self.conv_5 = nn.Conv2d(16 * nf              , 32 * nf , kernel_size=3 , stride=2   , padding=1)
        self.conv_6 = nn.Conv2d(32 * nf              , 32 * nf , kernel_size=3 , padding=1)

        # down
        self.norm_1 = nn.InstanceNorm2d( 2 * nf, affine=False)
        self.norm_2 = nn.InstanceNorm2d( 4 * nf, affine=False)
        self.norm_3 = nn.InstanceNorm2d( 8 * nf, affine=False)
        self.norm_4 = nn.InstanceNorm2d(16 * nf, affine=False)
        self.norm_5 = nn.InstanceNorm2d(32 * nf, affine=False)

        # up
        opt_copy = argparse.Namespace(**vars(opt))
        opt_copy.semantic_nc = opt.semantic_nc + 2048
        self.spaderesblk_6 = SPADEResnetBlock(32 * nf, 32 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 1024
        self.spaderesblk_5 = SPADEResnetBlock(32 * nf, 16 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 512
        self.spaderesblk_4 = SPADEResnetBlock(16 * nf,  8 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 256
        self.spaderesblk_3 = SPADEResnetBlock( 8 * nf,  4 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 128
        self.spaderesblk_2 = SPADEResnetBlock( 4 * nf,  2 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 64
        self.spaderesblk_1 = SPADEResnetBlock( 2 * nf,  1 * nf, opt_copy)

        # up
        self.to_rgb_6 = ToRGB(32 * nf)
        self.to_rgb_5 = ToRGB(16 * nf)
        self.to_rgb_4 = ToRGB( 8 * nf)
        self.to_rgb_3 = ToRGB( 4 * nf)
        self.to_rgb_2 = ToRGB( 2 * nf)
        self.to_rgb_1 = ToRGB( 1 * nf)

    def forward(self, input, z=None):
        latent_0 = self.conv_0(input)                             # 64   # 512
        latent_1 = self.norm_1(self.conv_1(self.lrelu(latent_0))) # 128  # 256
        latent_2 = self.norm_2(self.conv_2(self.lrelu(latent_1))) # 256  # 128
        latent_3 = self.norm_3(self.conv_3(self.lrelu(latent_2))) # 512  # 64
        latent_4 = self.norm_4(self.conv_4(self.lrelu(latent_3))) # 1024 # 32
        latent_5 = self.norm_5(self.conv_5(self.lrelu(latent_4))) # 2048 # 16x

        x = self.conv_6(self.lrelu(latent_5))      # 2048 # 16x

        x = self.spaderesblk_6(x, input, latent_5) # 2048 # 16x
        out = self.to_rgb_6(x)    # 3    # 16x
        out = self.up(out)        # 3    # 32x
        x = self.up(x)            # 2048 # 32x 
        x = self.spaderesblk_5(x, input, latent_4) # 1024 # 32
        out = out + self.to_rgb_5(x) # 3 # 32
        out = self.up(out)      # 3 # 64
        x = self.up(x)
        x = self.spaderesblk_4(x, input, latent_3) # 512  # 64
        out = out + self.to_rgb_4(x) # 3 # 64
        out = self.up(out)      # 3 # 128
        x = self.up(x)
        x = self.spaderesblk_3(x, input, latent_2) # 256  # 128
        out = out + self.to_rgb_3(x) # 3 # 128
        out = self.up(out)      # 3 # 256
        x = self.up(x)
        x = self.spaderesblk_2(x, input, latent_1) # 128  # 256
        out = out + self.to_rgb_2(x) # 3 # 256
        out = self.up(out)      # 3 # 512
        x = self.up(x)
        x = self.spaderesblk_1(x, input, latent_0) # 64   # 512
        out = out + self.to_rgb_1(x) # 3 # 512
        return out

class MiGANSkipLatentALLGenerator(BaseNetwork):
    """
    まずはJPUを使わずStyleGAN2のskip connectionとResidual netsが効くか確かめる．
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf #64

        # sharing modules in each hierarchy, number of learning parameters == 0
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.up = nn.Upsample(scale_factor=2, mode=opt.resize_mode, align_corners=opt.resize_align_corners)

        # down
        self.conv_0 = nn.Conv2d(self.opt.semantic_nc , 1  * nf , kernel_size=3 , stride=1   , padding=1)
        self.conv_1 = nn.Conv2d(1  * nf              , 2  * nf , kernel_size=3 , stride=2   , padding=1)
        self.conv_2 = nn.Conv2d(2  * nf              , 4  * nf , kernel_size=3 , stride=2   , padding=1)
        self.conv_3 = nn.Conv2d(4  * nf              , 8  * nf , kernel_size=3 , stride=2   , padding=1)
        self.conv_4 = nn.Conv2d(8  * nf              , 16 * nf , kernel_size=3 , stride=2   , padding=1)
        self.conv_5 = nn.Conv2d(16 * nf              , 16 * nf , kernel_size=3 , padding=1)
        self.conv_6 = nn.Conv2d(16 * nf              , 16 * nf , kernel_size=3 , padding=1)

        # down
        self.norm_1 = nn.InstanceNorm2d( 2 * nf, affine=False)
        self.norm_2 = nn.InstanceNorm2d( 4 * nf, affine=False)
        self.norm_3 = nn.InstanceNorm2d( 8 * nf, affine=False)
        self.norm_4 = nn.InstanceNorm2d(16 * nf, affine=False)
        self.norm_5 = nn.InstanceNorm2d(16 * nf, affine=False)

        # up
        opt_copy = argparse.Namespace(**vars(opt))
        opt_copy.semantic_nc = opt.semantic_nc + 1024
        self.spaderesblk_6 = SPADEResnetBlock(16 * nf, 16 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 1024
        self.spaderesblk_5 = SPADEResnetBlock(16 * nf, 16 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 512
        self.spaderesblk_4 = SPADEResnetBlock(16 * nf,  8 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 256
        self.spaderesblk_3 = SPADEResnetBlock( 8 * nf,  4 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 128
        self.spaderesblk_2 = SPADEResnetBlock( 4 * nf,  2 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 64
        self.spaderesblk_1 = SPADEResnetBlock( 2 * nf,  1 * nf, opt_copy)

        # up
        self.to_rgb_6 = ToRGB(16 * nf)
        self.to_rgb_5 = ToRGB(16 * nf)
        self.to_rgb_4 = ToRGB( 8 * nf)
        self.to_rgb_3 = ToRGB( 4 * nf)
        self.to_rgb_2 = ToRGB( 2 * nf)
        self.to_rgb_1 = ToRGB( 1 * nf)

    def forward(self, input, z=None):
        latent_0 = self.conv_0(input)                             # 64   # 512
        latent_1 = self.norm_1(self.conv_1(self.lrelu(latent_0))) # 128  # 256
        latent_2 = self.norm_2(self.conv_2(self.lrelu(latent_1))) # 256  # 128
        latent_3 = self.norm_3(self.conv_3(self.lrelu(latent_2))) # 512  # 64
        latent_4 = self.norm_4(self.conv_4(self.lrelu(latent_3))) # 1024 # 32
        latent_5 = self.norm_5(self.conv_5(self.lrelu(latent_4))) # 1024 # 32

        x = self.conv_6(self.lrelu(latent_5))      # 1024 # 32
        x = self.spaderesblk_6(x, input, latent_5) # 1024 # 32
        out = self.to_rgb_6(x)       # 3 # 32
        x = self.spaderesblk_5(x, input, latent_4) # 1024 # 32
        out = out + self.to_rgb_5(x) # 3 # 32
        out = self.up(out)      # 3 # 64
        x = self.up(x)
        x = self.spaderesblk_4(x, input, latent_3) # 512  # 64
        out = out + self.to_rgb_4(x) # 3 # 64
        out = self.up(out)      # 3 # 128
        x = self.up(x)
        x = self.spaderesblk_3(x, input, latent_2) # 256  # 128
        out = out + self.to_rgb_3(x) # 3 # 128
        out = self.up(out)      # 3 # 256
        x = self.up(x)
        x = self.spaderesblk_2(x, input, latent_1) # 128  # 256
        out = out + self.to_rgb_2(x) # 3 # 256
        out = self.up(out)      # 3 # 512
        x = self.up(x)
        x = self.spaderesblk_1(x, input, latent_0) # 64   # 512
        out = out + self.to_rgb_1(x) # 3 # 512
        return out

class MiGANLatentALLBNGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf #64

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        # bottleneck
        if self.opt.use_vae:
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            self.conv_6 = nn.Conv2d(16 * nf, 16 * nf, kernel_size=3, padding=1)

        # sharing modules in each hierarchy, number of learning parameters == 0
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.up = nn.Upsample(scale_factor=2)

        # down
        self.conv_0 = nn.Conv2d(self.opt.semantic_nc, 1  * nf, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(1  * nf             , 2  * nf, kernel_size=3, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(2  * nf             , 4  * nf, kernel_size=3, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(4  * nf             , 8  * nf, kernel_size=3, stride=2, padding=1)
        self.conv_4 = nn.Conv2d(8  * nf             , 16 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_5 = nn.Conv2d(16 * nf             , 16 * nf, kernel_size=3, padding=1)

        # down
        self.norm_1 = SynchronizedBatchNorm2d( 2 * nf, affine=True)
        self.norm_2 = SynchronizedBatchNorm2d( 4 * nf, affine=True)
        self.norm_3 = SynchronizedBatchNorm2d( 8 * nf, affine=True)
        self.norm_4 = SynchronizedBatchNorm2d(16 * nf, affine=True)
        self.norm_5 = SynchronizedBatchNorm2d(16 * nf, affine=True)

        # up
        opt_copy = argparse.Namespace(**vars(opt))
        opt_copy.semantic_nc = opt.semantic_nc + 1024
        self.spaderesblk_6 = SPADEResnetBlock(16 * nf, 16 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 1024
        self.spaderesblk_5 = SPADEResnetBlock(16 * nf, 16 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 512
        self.spaderesblk_4 = SPADEResnetBlock(16 * nf,  8 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 256
        self.spaderesblk_3 = SPADEResnetBlock( 8 * nf,  4 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 128
        self.spaderesblk_2 = SPADEResnetBlock( 4 * nf,  2 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 64
        self.spaderesblk_1 = SPADEResnetBlock( 2 * nf,  1 * nf, opt_copy)

        self.conv_img = nn.Conv2d(1 * nf , 3, kernel_size=3, padding=1)

    def compute_latent_vector_size(self, opt):
        sw, sh = 16, 16
        return sw, sh

    def forward(self, input, z=None):
        latent_0 = self.conv_0(input)                             # 64   # 256
        latent_1 = self.norm_1(self.conv_1(self.lrelu(latent_0))) # 128  # 128
        latent_2 = self.norm_2(self.conv_2(self.lrelu(latent_1))) # 256  # 64
        latent_3 = self.norm_3(self.conv_3(self.lrelu(latent_2))) # 512  # 32
        latent_4 = self.norm_4(self.conv_4(self.lrelu(latent_3))) # 1024 # 16
        latent_5 = self.norm_5(self.conv_5(self.lrelu(latent_4))) # 1024 # 16

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            x = self.conv_6(self.lrelu(latent_5)) # 1024 # 16

        x = self.spaderesblk_6(x, input, latent_5) # 1024 # 16
        x = self.spaderesblk_5(x, input, latent_4) # 1024 # 16
        x = self.up(x)
        x = self.spaderesblk_4(x, input, latent_3) # 512  # 32
        x = self.up(x)
        x = self.spaderesblk_3(x, input, latent_2) # 256  # 64
        x = self.up(x)
        x = self.spaderesblk_2(x, input, latent_1) # 128  # 128
        x = self.up(x)
        x = self.spaderesblk_1(x, input, latent_0) #)     # 64   # 256
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        return x

class MiGANLatentALLGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf #64

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        # bottleneck
        if self.opt.use_vae:
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            self.conv_6 = nn.Conv2d(16 * nf, 16 * nf, kernel_size=3, padding=1)

        # sharing modules in each hierarchy, number of learning parameters == 0
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.up = nn.Upsample(scale_factor=2)

        # down
        self.conv_0 = nn.Conv2d(self.opt.semantic_nc, 1  * nf, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(1  * nf             , 2  * nf, kernel_size=3, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(2  * nf             , 4  * nf, kernel_size=3, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(4  * nf             , 8  * nf, kernel_size=3, stride=2, padding=1)
        self.conv_4 = nn.Conv2d(8  * nf             , 16 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_5 = nn.Conv2d(16 * nf             , 16 * nf, kernel_size=3, padding=1)

        # down
        self.norm_1 = nn.InstanceNorm2d( 2 * nf, affine=False)
        self.norm_2 = nn.InstanceNorm2d( 4 * nf, affine=False)
        self.norm_3 = nn.InstanceNorm2d( 8 * nf, affine=False)
        self.norm_4 = nn.InstanceNorm2d(16 * nf, affine=False)
        self.norm_5 = nn.InstanceNorm2d(16 * nf, affine=False)

        # up
        opt_copy = argparse.Namespace(**vars(opt))
        opt_copy.semantic_nc = opt.semantic_nc + 1024
        self.spaderesblk_6 = SPADEResnetBlock(16 * nf, 16 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 1024
        self.spaderesblk_5 = SPADEResnetBlock(16 * nf, 16 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 512
        self.spaderesblk_4 = SPADEResnetBlock(16 * nf,  8 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 256
        self.spaderesblk_3 = SPADEResnetBlock( 8 * nf,  4 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 128
        self.spaderesblk_2 = SPADEResnetBlock( 4 * nf,  2 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 64
        self.spaderesblk_1 = SPADEResnetBlock( 2 * nf,  1 * nf, opt_copy)

        self.conv_img = nn.Conv2d(1 * nf , 3, kernel_size=3, padding=1)

    def compute_latent_vector_size(self, opt):
        sw, sh = 16, 16
        return sw, sh

    def forward(self, input, z=None):
        latent_0 = self.conv_0(input)                             # 64   # 256
        latent_1 = self.norm_1(self.conv_1(self.lrelu(latent_0))) # 128  # 128
        latent_2 = self.norm_2(self.conv_2(self.lrelu(latent_1))) # 256  # 64
        latent_3 = self.norm_3(self.conv_3(self.lrelu(latent_2))) # 512  # 32
        latent_4 = self.norm_4(self.conv_4(self.lrelu(latent_3))) # 1024 # 16
        latent_5 = self.norm_5(self.conv_5(self.lrelu(latent_4))) # 1024 # 16

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            x = self.conv_6(self.lrelu(latent_5)) # 1024 # 16

        x = self.spaderesblk_6(x, input, latent_5) # 1024 # 16
        x = self.spaderesblk_5(x, input, latent_4) # 1024 # 16
        x = self.up(x)
        x = self.spaderesblk_4(x, input, latent_3) # 512  # 32
        x = self.up(x)
        x = self.spaderesblk_3(x, input, latent_2) # 256  # 64
        x = self.up(x)
        x = self.spaderesblk_2(x, input, latent_1) # 128  # 128
        x = self.up(x)
        x = self.spaderesblk_1(x, input, latent_0) #)     # 64   # 256
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        return x

class MaGANResidualGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
#        parser.set_defaults(norm_G='instance')

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf #64

        # sharing modules in each hierarchy, number of learning parameters == 0
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.up = nn.Upsample(scale_factor=2)

        # down
        self.conv_0 = nn.Conv2d(184    ,  1 * nf, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(1  * nf,  2 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(2  * nf,  4 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(4  * nf,  8 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_4 = nn.Conv2d(8  * nf, 16 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_5 = nn.Conv2d(16 * nf, 16 * nf, kernel_size=3, padding=1)
        self.conv_6 = nn.Conv2d(16 * nf, 16 * nf, kernel_size=3, padding=1)

        # down
        #self.norm_0 = nn.InstanceNorm2d( 1 * nf, affine=False)
        self.norm_1 = nn.InstanceNorm2d( 2 * nf, affine=False)
        self.norm_2 = nn.InstanceNorm2d( 4 * nf, affine=False)
        self.norm_3 = nn.InstanceNorm2d( 8 * nf, affine=False)
        self.norm_4 = nn.InstanceNorm2d(16 * nf, affine=False)
        self.norm_5 = nn.InstanceNorm2d(16 * nf, affine=False)
        self.norm_6 = nn.InstanceNorm2d(16 * nf, affine=False)

        # res
        self.res1 = ResnetBlock(16 * nf, norm_layer=nn.InstanceNorm2d, kernel_size=2)
        self.res2 = ResnetBlock(16 * nf, norm_layer=nn.InstanceNorm2d, kernel_size=2)
        self.res3 = ResnetBlock(16 * nf, norm_layer=nn.InstanceNorm2d, kernel_size=2)
        self.res4 = ResnetBlock(16 * nf, norm_layer=nn.InstanceNorm2d, kernel_size=2)
        self.res5 = ResnetBlock(16 * nf, norm_layer=nn.InstanceNorm2d, kernel_size=2)
        self.res6 = ResnetBlock(16 * nf, norm_layer=nn.InstanceNorm2d, kernel_size=2)
        self.res7 = ResnetBlock(16 * nf, norm_layer=nn.InstanceNorm2d, kernel_size=2)
        self.res8 = ResnetBlock(16 * nf, norm_layer=nn.InstanceNorm2d, kernel_size=2)

        # up
        opt_copy = argparse.Namespace(**vars(opt))
        opt_copy.semantic_nc = opt.semantic_nc + 1024
        self.spaderesblk_6 = SPADEResnetBlock(16 * nf, 16 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 1024
        self.spaderesblk_5 = SPADEResnetBlock(16 * nf, 16 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 512
        self.spaderesblk_4 = SPADEResnetBlock(16 * nf,  8 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 256
        self.spaderesblk_3 = SPADEResnetBlock( 8 * nf,  4 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 128
        self.spaderesblk_2 = SPADEResnetBlock( 4 * nf,  2 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 64
        self.spaderesblk_1 = SPADEResnetBlock( 2 * nf,  1 * nf, opt_copy)

        self.conv_img = nn.Conv2d(1 * nf , 3, kernel_size=3, padding=1)

    def forward(self, input, z=None):
        latent_0 = self.conv_0(input)                              # 64   # 256
        latent_1 = self.norm_1(self.conv_1(self.lrelu(latent_0))) # 128  # 128
        latent_2 = self.norm_2(self.conv_2(self.lrelu(latent_1))) # 256  # 64
        latent_3 = self.norm_3(self.conv_3(self.lrelu(latent_2))) # 512  # 32
        latent_4 = self.norm_4(self.conv_4(self.lrelu(latent_3))) # 1024 # 16
        latent_5 = self.norm_5(self.conv_5(self.lrelu(latent_4))) # 1024 # 16
        latent_6 = self.norm_6(self.conv_6(self.lrelu(latent_5))) # 1024 # 16
        x = latent_6
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = self.spaderesblk_6(x, input, latent_5) # 1024 # 16
        x = self.spaderesblk_5(x, input, latent_4) # 1024 # 16
        x = self.up(x)
        x = self.spaderesblk_4(x, input, latent_3) # 512  # 32
        x = self.up(x)
        x = self.spaderesblk_3(x, input, latent_2) # 256  # 64
        x = self.up(x)
        x = self.spaderesblk_2(x, input, latent_1) # 128  # 128
        x = self.up(x)
        x = self.spaderesblk_1(x, input, latent_0) #)     # 64   # 256
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        return x

class MaGANLatentALLGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
#        parser.set_defaults(norm_G='instance')

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf #64

        # sharing modules in each hierarchy, number of learning parameters == 0
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.up = nn.Upsample(scale_factor=2)

        # down
        self.conv_0 = nn.Conv2d(184    ,  1 * nf, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(1  * nf,  2 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(2  * nf,  4 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(4  * nf,  8 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_4 = nn.Conv2d(8  * nf, 16 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_5 = nn.Conv2d(16 * nf, 16 * nf, kernel_size=3, padding=1)
        self.conv_6 = nn.Conv2d(16 * nf, 16 * nf, kernel_size=3, padding=1)

        # down
        #self.norm_0 = nn.InstanceNorm2d( 1 * nf, affine=False)
        self.norm_1 = nn.InstanceNorm2d( 2 * nf, affine=False)
        self.norm_2 = nn.InstanceNorm2d( 4 * nf, affine=False)
        self.norm_3 = nn.InstanceNorm2d( 8 * nf, affine=False)
        self.norm_4 = nn.InstanceNorm2d(16 * nf, affine=False)
        self.norm_5 = nn.InstanceNorm2d(16 * nf, affine=False)
        self.norm_6 = lambda x:x

        # res
        #self.res1 = ResnetBlock(opt.ngf * mult, norm_layer=norm_layer, activation=activation, kernel_size=opt.resnet_kernel_size)

        # up
        opt_copy = argparse.Namespace(**vars(opt))
        opt_copy.semantic_nc = opt.semantic_nc + 1024
        self.spaderesblk_6 = SPADEResnetBlock(16 * nf, 16 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 1024
        self.spaderesblk_5 = SPADEResnetBlock(16 * nf, 16 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 512
        self.spaderesblk_4 = SPADEResnetBlock(16 * nf,  8 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 256
        self.spaderesblk_3 = SPADEResnetBlock( 8 * nf,  4 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 128
        self.spaderesblk_2 = SPADEResnetBlock( 4 * nf,  2 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc + 64
        self.spaderesblk_1 = SPADEResnetBlock( 2 * nf,  1 * nf, opt_copy)

        self.conv_img = nn.Conv2d(1 * nf , 3, kernel_size=3, padding=1)

    def forward(self, input, z=None):
        latent_0 = self.conv_0(input)                              # 64   # 256
        latent_1 = self.norm_1(self.conv_1(self.lrelu(latent_0))) # 128  # 128
        latent_2 = self.norm_2(self.conv_2(self.lrelu(latent_1))) # 256  # 64
        latent_3 = self.norm_3(self.conv_3(self.lrelu(latent_2))) # 512  # 32
        latent_4 = self.norm_4(self.conv_4(self.lrelu(latent_3))) # 1024 # 16
        latent_5 = self.norm_5(self.conv_5(self.lrelu(latent_4))) # 1024 # 16
        latent_6 = self.norm_6(self.conv_6(self.lrelu(latent_5))) # 1024 # 16
        x = latent_6
        x = self.spaderesblk_6(x, input, latent_5) # 1024 # 16
        x = self.spaderesblk_5(x, input, latent_4) # 1024 # 16
        x = self.up(x)
        x = self.spaderesblk_4(x, input, latent_3) # 512  # 32
        x = self.up(x)
        x = self.spaderesblk_3(x, input, latent_2) # 256  # 64
        x = self.up(x)
        x = self.spaderesblk_2(x, input, latent_1) # 128  # 128
        x = self.up(x)
        x = self.spaderesblk_1(x, input, latent_0) #)     # 64   # 256
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        return x

class MaGANResALLInputGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
#        parser.set_defaults(norm_G='instance')

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf #64

        # sharing modules in each hierarchy, number of learning parameters == 0
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.up = nn.Upsample(scale_factor=2)

        # down
        self.conv_0 = nn.Conv2d(184    ,  1 * nf, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(1  * nf,  2 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(2  * nf,  4 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(4  * nf,  8 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_4 = nn.Conv2d(8  * nf, 16 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_5 = nn.Conv2d(16 * nf, 16 * nf, kernel_size=3, padding=1)
        self.conv_6 = nn.Conv2d(16 * nf, 16 * nf, kernel_size=3, padding=1)

        # down
        #self.norm_0 = nn.InstanceNorm2d( 1 * nf, affine=False)
        self.norm_1 = nn.InstanceNorm2d( 2 * nf, affine=False)
        self.norm_2 = nn.InstanceNorm2d( 4 * nf, affine=False)
        self.norm_3 = nn.InstanceNorm2d( 8 * nf, affine=False)
        self.norm_4 = nn.InstanceNorm2d(16 * nf, affine=False)
        self.norm_5 = nn.InstanceNorm2d(16 * nf, affine=False)
        self.norm_6 = lambda x:x

        # res
        #self.res1 = ResnetBlock(opt.ngf * mult, norm_layer=norm_layer, activation=activation, kernel_size=opt.resnet_kernel_size)

        # up
        opt_copy = argparse.Namespace(**vars(opt))
        opt_copy.semantic_nc = opt.semantic_nc -1 + 1024
        self.spaderesblk_6 = SPADEResnetBlock(16 * nf, 16 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc -1 + 1024
        self.spaderesblk_5 = SPADEResnetBlock(16 * nf, 16 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc -1 + 512
        self.spaderesblk_4 = SPADEResnetBlock(16 * nf,  8 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc -1 + 256
        self.spaderesblk_3 = SPADEResnetBlock( 8 * nf,  4 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc -1 + 128
        self.spaderesblk_2 = SPADEResnetBlock( 4 * nf,  2 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc -1 + 1
        self.spaderesblk_1 = SPADEResnetBlock( 2 * nf,  1 * nf, opt_copy)

        self.conv_img = nn.Conv2d(1 * nf , 3, kernel_size=3, padding=1)

    def forward(self, input, z=None):
        edge = input[:,-1:,:,:]
        seg = input[:,:-1,:,:]
        latent_0 = self.conv_0(input)                              # 64   # 256
        latent_1 = self.norm_1(self.conv_1(self.lrelu(latent_0))) # 128  # 128
        latent_2 = self.norm_2(self.conv_2(self.lrelu(latent_1))) # 256  # 64
        latent_3 = self.norm_3(self.conv_3(self.lrelu(latent_2))) # 512  # 32
        latent_4 = self.norm_4(self.conv_4(self.lrelu(latent_3))) # 1024 # 16
        latent_5 = self.norm_5(self.conv_5(self.lrelu(latent_4))) # 1024 # 16
        latent_6 = self.norm_6(self.conv_6(self.lrelu(latent_5))) # 1024 # 16
        x = latent_6
        x = self.spaderesblk_6(x, seg, latent_5) # 1024 # 16
        x = self.spaderesblk_5(x, seg, latent_4) # 1024 # 16
        x = self.up(x)
        x = self.spaderesblk_4(x, seg, latent_3) # 512  # 32
        x = self.up(x)
        x = self.spaderesblk_3(x, seg, latent_2) # 256  # 64
        x = self.up(x)
        x = self.spaderesblk_2(x, seg, latent_1) # 128  # 128
        x = self.up(x)
        x = self.spaderesblk_1(x, seg, edge)     # 64   # 256
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        return x


class MaGANResInputGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
#        parser.set_defaults(norm_G='instance')

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf #64

        # sharing modules in each hierarchy, number of learning parameters == 0
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.up = nn.Upsample(scale_factor=2)

        # down
        self.conv_0 = nn.Conv2d(1      ,  1 * nf, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(1  * nf,  2 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(2  * nf,  4 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(4  * nf,  8 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_4 = nn.Conv2d(8  * nf, 16 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_5 = nn.Conv2d(16 * nf, 16 * nf, kernel_size=3, padding=1)
        self.conv_6 = nn.Conv2d(16 * nf, 16 * nf, kernel_size=3, padding=1)

        # down
        #self.norm_0 = nn.InstanceNorm2d( 1 * nf, affine=False)
        self.norm_1 = nn.InstanceNorm2d( 2 * nf, affine=False)
        self.norm_2 = nn.InstanceNorm2d( 4 * nf, affine=False)
        self.norm_3 = nn.InstanceNorm2d( 8 * nf, affine=False)
        self.norm_4 = nn.InstanceNorm2d(16 * nf, affine=False)
        self.norm_5 = nn.InstanceNorm2d(16 * nf, affine=False)
        self.norm_6 = lambda x:x

        # res
        #self.res1 = ResnetBlock(opt.ngf * mult, norm_layer=norm_layer, activation=activation, kernel_size=opt.resnet_kernel_size)

        # up
        opt_copy = argparse.Namespace(**vars(opt))
        opt_copy.semantic_nc = opt.semantic_nc -1 + 1024
        self.spaderesblk_6 = SPADEResnetBlock(16 * nf, 16 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc -1 + 1024
        self.spaderesblk_5 = SPADEResnetBlock(16 * nf, 16 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc -1 + 512
        self.spaderesblk_4 = SPADEResnetBlock(16 * nf,  8 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc -1 + 256
        self.spaderesblk_3 = SPADEResnetBlock( 8 * nf,  4 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc -1 + 128
        self.spaderesblk_2 = SPADEResnetBlock( 4 * nf,  2 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc -1 + 1
        self.spaderesblk_1 = SPADEResnetBlock( 2 * nf,  1 * nf, opt_copy)

        self.conv_img = nn.Conv2d(1 * nf , 3, kernel_size=3, padding=1)

    def forward(self, input, z=None):
        edge = input[:,-1:,:,:]
        seg = input[:,:-1,:,:]
        latent_0 = self.conv_0(edge)                              # 64   # 256
        latent_1 = self.norm_1(self.conv_1(self.lrelu(latent_0))) # 128  # 128
        latent_2 = self.norm_2(self.conv_2(self.lrelu(latent_1))) # 256  # 64
        latent_3 = self.norm_3(self.conv_3(self.lrelu(latent_2))) # 512  # 32
        latent_4 = self.norm_4(self.conv_4(self.lrelu(latent_3))) # 1024 # 16
        latent_5 = self.norm_5(self.conv_5(self.lrelu(latent_4))) # 1024 # 16
        latent_6 = self.norm_6(self.conv_6(self.lrelu(latent_5))) # 1024 # 16
        x = latent_6
        x = self.spaderesblk_6(x, seg, latent_5) # 1024 # 16
        x = self.spaderesblk_5(x, seg, latent_4) # 1024 # 16
        x = self.up(x)
        x = self.spaderesblk_4(x, seg, latent_3) # 512  # 32
        x = self.up(x)
        x = self.spaderesblk_3(x, seg, latent_2) # 256  # 64
        x = self.up(x)
        x = self.spaderesblk_2(x, seg, latent_1) # 128  # 128
        x = self.up(x)
        x = self.spaderesblk_1(x, seg, edge)     # 64   # 256
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        return x

class MaGANResALLInputGeneratorV2(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
#        parser.set_defaults(norm_G='instance')

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf #64

        # sharing modules in each hierarchy, number of learning parameters == 0
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.up = nn.Upsample(scale_factor=2)

        # down
        self.conv_0 = nn.Conv2d(184    ,  1 * nf, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(1  * nf,  2 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(2  * nf,  4 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(4  * nf,  8 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_4 = nn.Conv2d(8  * nf, 16 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_5 = nn.Conv2d(16 * nf, 16 * nf, kernel_size=3, padding=1)
        self.conv_6 = nn.Conv2d(16 * nf, 16 * nf, kernel_size=3, padding=1)

        # down
        #self.norm_0 = nn.InstanceNorm2d( 1 * nf, affine=False)
        self.norm_1 = nn.InstanceNorm2d( 2 * nf, affine=False)
        self.norm_2 = nn.InstanceNorm2d( 4 * nf, affine=False)
        self.norm_3 = nn.InstanceNorm2d( 8 * nf, affine=False)
        self.norm_4 = nn.InstanceNorm2d(16 * nf, affine=False)
        self.norm_5 = nn.InstanceNorm2d(16 * nf, affine=False)
        self.norm_6 = lambda x:x

        # res
        #self.res1 = ResnetBlock(opt.ngf * mult, norm_layer=norm_layer, activation=activation, kernel_size=opt.resnet_kernel_size)

        # up
        opt_copy = argparse.Namespace(**vars(opt))
        opt_copy.semantic_nc = opt.semantic_nc -1 + 1024
        self.spaderesblk_6 = SPADEResnetBlock(16 * nf, 16 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc -1 + 1024
        self.spaderesblk_5 = SPADEResnetBlock(16 * nf, 16 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc -1 + 512
        self.spaderesblk_4 = SPADEResnetBlock(16 * nf,  8 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc -1 + 256
        self.spaderesblk_3 = SPADEResnetBlock( 8 * nf,  4 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc -1 + 128
        self.spaderesblk_2 = SPADEResnetBlock( 4 * nf,  2 * nf, opt_copy)
        opt_copy.semantic_nc = opt.semantic_nc -1 + 64
        self.spaderesblk_1 = SPADEResnetBlock( 2 * nf,  1 * nf, opt_copy)

        self.conv_img = nn.Conv2d(1 * nf , 3, kernel_size=3, padding=1)

    def forward(self, input, z=None):
        edge = input[:,-1:,:,:]
        seg = input[:,:-1,:,:]
        latent_0 = self.conv_0(input)                              # 64   # 256
        latent_1 = self.norm_1(self.conv_1(self.lrelu(latent_0))) # 128  # 128
        latent_2 = self.norm_2(self.conv_2(self.lrelu(latent_1))) # 256  # 64
        latent_3 = self.norm_3(self.conv_3(self.lrelu(latent_2))) # 512  # 32
        latent_4 = self.norm_4(self.conv_4(self.lrelu(latent_3))) # 1024 # 16
        latent_5 = self.norm_5(self.conv_5(self.lrelu(latent_4))) # 1024 # 16
        latent_6 = self.norm_6(self.conv_6(self.lrelu(latent_5))) # 1024 # 16
        x = latent_6
        x = self.spaderesblk_6(x, seg, latent_5) # 1024 # 16
        x = self.spaderesblk_5(x, seg, latent_4) # 1024 # 16
        x = self.up(x)
        x = self.spaderesblk_4(x, seg, latent_3) # 512  # 32
        x = self.up(x)
        x = self.spaderesblk_3(x, seg, latent_2) # 256  # 64
        x = self.up(x)
        x = self.spaderesblk_2(x, seg, latent_1) # 128  # 128
        x = self.up(x)
        x = self.spaderesblk_1(x, seg, latent_0)     # 64   # 256
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        return x

class MaGANGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
#        parser.set_defaults(norm_G='instance')

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf #64

        # sharing modules in each hierarchy, number of learning parameters == 0
        self.lrelu = nn.LeakyReLU(0.2, True)

        # down
        self.conv_0 = nn.Conv2d(1     ,  1 * nf, kernel_size=3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(1 * nf,  2 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(2 * nf,  4 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(4 * nf,  8 * nf, kernel_size=3, stride=2, padding=1)
        self.conv_4 = nn.Conv2d(8 * nf, 16 * nf, kernel_size=3, stride=2, padding=1)

        # down
        #self.norm_0 = nn.InstanceNorm2d( 1 * nf, affine=False)
        self.norm_1 = nn.InstanceNorm2d( 2 * nf, affine=False)
        self.norm_2 = nn.InstanceNorm2d( 4 * nf, affine=False)
        self.norm_3 = nn.InstanceNorm2d( 8 * nf, affine=False)
        #self.norm_4 = nn.InstanceNorm2d(16 * nf, affine=False)

        # up
        opt_copy = argparse.Namespace(**vars(opt))
        opt_copy.semantic_nc -= 1
        self.spaderesblk_5 = SPADEResnetBlock(16 * nf, 16 * nf, opt_copy)
        self.spaderesblk_4 = SPADEResnetBlock(16 * nf,  8 * nf, opt_copy)
        self.spaderesblk_3 = SPADEResnetBlock( 8 * nf,  4 * nf, opt_copy)
        self.spaderesblk_2 = SPADEResnetBlock( 4 * nf,  2 * nf, opt_copy)
        self.spaderesblk_1 = SPADEResnetBlock( 2 * nf,  1 * nf, opt_copy)

        # up
        self.convtr_4 = nn.ConvTranspose2d(32 * nf, 16 * nf, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtr_3 = nn.ConvTranspose2d(16 * nf,  8 * nf, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtr_2 = nn.ConvTranspose2d( 8 * nf,  4 * nf, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtr_1 = nn.ConvTranspose2d( 4 * nf,  2 * nf, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.conv_img = nn.Conv2d(2 * nf , 3, kernel_size=3, padding=1)

    def forward(self, input, z=None):
        edge = input[:,-1:,:,:]                                   # 1    # 256
        seg = input[:,:-1,:,:]                                    # 183  # 256
#        seg = input                                              # 184  # 256
        latent_0 = self.conv_0(edge)                              # 64   # 256
        latent_1 = self.norm_1(self.conv_1(self.lrelu(latent_0))) # 128  # 128
        latent_2 = self.norm_2(self.conv_2(self.lrelu(latent_1))) # 256  # 64
        latent_3 = self.norm_3(self.conv_3(self.lrelu(latent_2))) # 512  # 32
        latent_4 =             self.conv_4(self.lrelu(latent_3))  # 1024 # 16
        x = self.spaderesblk_5(latent_4, seg)                     # 1024 # 16
        x = torch.cat([x, latent_4], 1)                           # 2048 # 16
        x = self.convtr_4(x)                                      # 1024 # 32
        x = self.spaderesblk_4(x, seg)                            # 512  # 32
        x = torch.cat([x, latent_3], 1)                           # 1024 # 32
        x = self.convtr_3(x)                                      # 512  # 64
        x = self.spaderesblk_3(x, seg)                            # 256  # 64
        x = torch.cat([x, latent_2], 1)                           # 512  # 64
        x = self.convtr_2(x)                                      # 256  # 128
        x = self.spaderesblk_2(x, seg)                            # 128  # 128
        x = torch.cat([x, latent_1], 1)                           # 256  # 128
        x = self.convtr_1(x)                                      # 128  # 256
        x = self.spaderesblk_1(x, seg)                            # 64   # 256
        x = torch.cat([x, latent_0], 1)                           # 128  # 256
        x = self.conv_img(F.leaky_relu(x, 2e-1))                  # 3    # 256
        x = F.tanh(x)
        return x


class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        if opt.use_vae:
            # In case of VAE, we will sample from random z vector
            self.fc = nn.Linear(opt.z_dim, 16 * nf * self.sw * self.sh)
        else:
            # Otherwise, we make the network deterministic by starting with
            # downsampled segmentation map instead of random z
            self.fc = nn.Conv2d(self.opt.semantic_nc, 16 * nf, 3, padding=1)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        if opt.num_upsampling_layers == 'most':
            self.up_4 = SPADEResnetBlock(1 * nf, nf // 2, opt)
            final_nc = nf // 2

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def compute_latent_vector_size(self, opt):
        if opt.num_upsampling_layers == 'normal':
            num_up_layers = 5
        elif opt.num_upsampling_layers == 'more':
            num_up_layers = 6
        elif opt.num_upsampling_layers == 'most':
            num_up_layers = 7
        else:
            raise ValueError('opt.num_upsampling_layers [%s] not recognized' %
                             opt.num_upsampling_layers)

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, z=None):
        seg = input

        if self.opt.use_vae:
            # we sample z from unit normal and reshape the tensor
            if z is None:
                z = torch.randn(input.size(0), self.opt.z_dim,
                                dtype=torch.float32, device=input.get_device())
            x = self.fc(z)
            x = x.view(-1, 16 * self.opt.ngf, self.sh, self.sw)
        else:
            # we downsample segmap and run convolution
            x = F.interpolate(seg, size=(self.sh, self.sw))
            x = self.fc(x)

        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        if self.opt.num_upsampling_layers == 'more' or \
           self.opt.num_upsampling_layers == 'most':
            x = self.up(x)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)
        x = self.up(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        if self.opt.num_upsampling_layers == 'most':
            x = self.up(x)
            x = self.up_4(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)

        return x


class Pix2PixHDGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--resnet_n_downsample', type=int, default=4, help='number of downsampling layers in netG')
        parser.add_argument('--resnet_n_blocks', type=int, default=9, help='number of residual blocks in the global generator network')
        parser.add_argument('--resnet_kernel_size', type=int, default=3,
                            help='kernel size of the resnet block')
        parser.add_argument('--resnet_initial_kernel_size', type=int, default=7,
                            help='kernel size of the first convolution')
        parser.set_defaults(norm_G='instance')
        return parser

    def __init__(self, opt):
        super().__init__()
        input_nc = opt.label_nc + (1 if opt.contain_dontcare_label else 0) + (0 if opt.no_instance else 1)

        norm_layer = get_nonspade_norm_layer(opt, opt.norm_G)
        activation = nn.ReLU(False)

        model = []

        # initial conv
        model += [nn.ReflectionPad2d(opt.resnet_initial_kernel_size // 2),
                  norm_layer(nn.Conv2d(input_nc, opt.ngf,
                                       kernel_size=opt.resnet_initial_kernel_size,
                                       padding=0)),
                  activation]

        # downsample
        mult = 1
        for i in range(opt.resnet_n_downsample):
            model += [norm_layer(nn.Conv2d(opt.ngf * mult, opt.ngf * mult * 2,
                                           kernel_size=3, stride=2, padding=1)),
                      activation]
            mult *= 2

        # resnet blocks
        for i in range(opt.resnet_n_blocks):
            model += [ResnetBlock(opt.ngf * mult,
                                  norm_layer=norm_layer,
                                  activation=activation,
                                  kernel_size=opt.resnet_kernel_size)]

        # upsample
        for i in range(opt.resnet_n_downsample):
            nc_in = int(opt.ngf * mult)
            nc_out = int((opt.ngf * mult) / 2)
            model += [norm_layer(nn.ConvTranspose2d(nc_in, nc_out,
                                                    kernel_size=3, stride=2,
                                                    padding=1, output_padding=1)),
                      activation]
            mult = mult // 2

        # final output conv
        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(nc_out, opt.output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, z=None):
        return self.model(input)
