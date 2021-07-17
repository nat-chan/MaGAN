import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, norm_layer=SynchronizedBatchNorm2d):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, padding, dilation, groups=inplanes, bias=bias)
        self.bn = norm_layer(inplanes)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class FlexibleJPU(nn.Module):
    def __init__(
            self,
            fins=[64, 128, 256, 512, 1024, 2048],
            fout=512,
            dilations=[1, 2, 4, 8],
            mode='nearest',
            align_corners=None,
            norm=SynchronizedBatchNorm2d,
            actv=nn.ReLU(inplace=True),
        ):
        super().__init__()
        self.dilations = dilations
        for i, fin in enumerate(fins):
            self.add_module(
                f'conv{i}', 
                nn.Sequential(nn.Conv2d(fin, fout, 3, padding=1, bias=False), norm(fout), actv, nn.Upsample(scale_factor=2**i, mode=mode, align_corners=align_corners))
            )
        for i, dilation in enumerate(dilations):
            self.add_module(
                f'dilation{i}', 
                nn.Sequential(SeparableConv2d(len(fins)*fout, fout, kernel_size=3, padding=dilation, dilation=dilation, bias=False), norm(fout), actv)
            )
        self.pointwise = nn.Conv2d(len(dilations)*fout, fout, kernel_size=1, bias=False)
    def __getitem__(self, key):
        return getattr(self, key)
    def forward(self, *inputs):
        out = torch.cat([self[f'conv{i}'](input) for i, input in enumerate(inputs)], dim=1)
        out = torch.cat([self[f'dilation{i}'](out) for i, _ in enumerate(self.dilations)], dim=1)
        out = self.pointwise(out)
        out = F.tanh(out)
        return out
