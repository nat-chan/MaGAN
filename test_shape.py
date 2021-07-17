#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.networks.standard as AlacGAN
import models.networks.generator as generator
import models.networks.encoder as encoder
import models.networks.architecture as architecture
from options.test_options import TestOptions
from options.train_options import TrainOptions
from models.pix2pix_model import Pix2PixModel
import unittest
#%%
class MyTest(unittest.TestCase):
    def test_NetG(self):
        model = AlacGAN.NetG()
        sketch = torch.zeros(1,1,512,512)
        hint = torch.zeros(1,4,128,128)
        sketch_feat = torch.zeros(1,512,32,32)
        out = model(sketch, hint, sketch_feat)
        self.assertEqual(out.shape, (1, 3, 512, 512))

    def test_NetI(self):
        model = AlacGAN.NetI()
        images = torch.zeros(1,1,512,512) #白黒をexpandして3チャンネルにしてる
        out = model(images)
        self.assertEqual(out.shape, (1, 512, 32, 32))

    def test_AlacGANGenerator(self):
        model = generator.AlacGANGenerator(None)
        images = torch.zeros(1,1,512,512) 
        out = model(images)
        self.assertEqual(out.shape, (1, 3, 512, 512))

    def test_I2V(self):
        model = encoder.I2VEncoder(None)
        images = torch.zeros(1,3,224,224)
        out = model(images)
        self.assertEqual(out.shape, (1, 1539, 1, 1))

    def test_ModulatedConv2d(self):
        model = architecture.ModulatedConv2d(in_channel=64, out_channel=32, kernel_size=3, padding=1, style_dim=1539, demodulate=True, bias=True)
        images = torch.zeros(1,64,256,256)
        style = torch.zeros(1539)
        out = model(images, style)
        self.assertEqual(out.shape, (1, 32, 256, 256))

    def test_ModulatedMaGAN(self):
        opt = TrainOptions().parse('--conf=parameters/512_modulatedv2bladeGskip_nomonochrome_single.yml --conf2=arameters/debug.yml')
        model = Pix2PixModel(opt)
    
    def test_GodUGenerator(self):
        opt = TrainOptions().parse()
        model = generator.GodUGenerator(opt)
        images = torch.zeros(1,1,512,512) 
        out = model(images)
        self.assertEqual(out.shape, (1, 3, 512, 512))

if __name__ == "__main__":
    unittest.main()
    test = MyTest()
