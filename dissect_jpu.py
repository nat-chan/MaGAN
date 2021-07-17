#!/usr/bin/env python
# coding: utf-8

# %%
import numpy as np
from PIL import Image
import data
from contextlib import redirect_stdout
import os
import torch
from util import util
from models.pix2pix_model import Pix2PixModel
from options.test_options import TestOptions
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt
# %%
opt = TestOptions().parse(
"""
--conf=./parameters/512_jpumagan_nomonochrome_single.yml
--conf2=./parameters/test.yml
--dissect
--which_epoch 1
""".split())
model = Pix2PixModel(opt)
model.eval()
# %%
dataloader = data.create_dataloader(opt)
# %%
chunk = 100
inputs = []
outputs = []
for i, d in enumerate(dataloader):
    with torch.no_grad():
        output = model.netG(d['hed'])
    inputs.append(d['hed'].cpu().numpy().transpose(2,3,1,0)[:,:,:,0])
    outputs.append(output['seg'].cpu().numpy().transpose(2,3,1,0)[:,:,:,0])
    if i == chunk:break
# %%
def get_feat(outputs):
    vec = np.vstack(list(o.reshape(o.shape[0]*o.shape[1], o.shape[2]) for o in outputs))
    pca = PCA()
    pca.fit(vec)
    feats = list(pca.transform(o.reshape(o.shape[0]*o.shape[1], o.shape[2])).reshape(*o.shape) for o in outputs)
    return feats
# %%
feats = get_feat(outputs)
# %%
def multiplot(feat):
    X = 4
    Y = 3
    plt.rcParams["figure.dpi"] = 300
    fig, ax = plt.subplots(Y, X)
    for x in range(X):
        for y in range(Y):
            n = y*X+x
            plt.axes(ax[y,x])
            plt.imshow(feat[:,:,n])
            plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
            plt.xlabel(f'n={n}')
    fig.show()
# %%
multiplot(feats[1])
# %%
multiplot(outputs[1])
# %%