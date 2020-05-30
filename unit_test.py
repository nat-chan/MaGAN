#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.networks.standard as AlacGAN
import models.networks.generator as generator
import models.networks.encoder as encoder
from PIL import Image
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

if __name__ == "__main__":
    test = MyTest()
    test.test_AlacGANGenerator()
    print(test)