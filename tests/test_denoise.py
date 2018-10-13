# -*- coding: utf-8 -*-

from pyE17.io import h5read
import numpy as np 
import torch as th
import imageio
from skpr._ext import skpr_thnn
from skpr.inout import *
denoise = skpr_thnn.denoise

p = '/home/pelzphil/projects/bm3d-gpu/lena_20.png'

im = imageio.imread(p)
print im.shape
print im.dtype

i = th.from_numpy(im).unsqueeze(0).expand(2,im.shape[0],im.shape[1]).clone()
ic = i.cuda()
print ic.size()
print type(ic)
out = ic.clone()

print ic.size(), out.size()
print type(ic), type(out)
denoise(ic,out)

showPlot(im,True)
showPlot(out.cpu().float().numpy()[0],True)

print type(i)