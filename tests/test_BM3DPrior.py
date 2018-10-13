# -*- coding: utf-8 -*-
from pyE17.io import h5read
import numpy as np
import torch as th
import imageio
from skpr._ext import skpr_thnn
from skpr.inout import *
import skpr.nn.functional as F
from torch.autograd import Variable

denoise = skpr_thnn.denoise

p = '/home/pelzphil/projects/bm3d-gpu/lena_20.png'

im = imageio.imread(p)
print im.shape
print im.dtype
showPlot(im, True)

im1 = im + 1j * im

i = th.from_numpy(im).unsqueeze(0)
i1 = th.from_numpy(im1.astype(np.complex64))
print i.size()
print type(i)
ic = i.cuda()
ic1 = i1.cuda()
out = ic.clone().byte()

# denoise(ic.byte(),out)
ins = Variable(ic1, requires_grad=True)
out1 = F.bm3d_prior(ins, th.FloatTensor([1]))

print('out1.is_leaf %s' % out1.is_leaf)
print('out1.requires_grad %s' % out1.requires_grad)
print('out1.grad_fn %s' % out1.grad_fn)

showPlot(out.cpu().float().numpy()[0], True)

print type(i)
