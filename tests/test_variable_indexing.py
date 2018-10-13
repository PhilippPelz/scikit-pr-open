# -*- coding: utf-8 -*-

from pyE17.io import h5read
import numpy as np
import torch as th
import imageio
from skpr._ext import skpr_thnn
from skpr.inout import *
import skpr.nn.functional as F
from torch.autograd import Variable
import skpr.nn.modules as M
import skpr.inout as io
import skpr.nn as p
from skpr.simulation.probe import focused_probe
import skpr.util as u
from skpr.optim import cg

x = np.array([1.3,5.5,10.1],dtype=np.float32)
y = np.array([0,0,0])
z = x #+ 1j* y

a = Variable(th.from_numpy(z), requires_grad=True)

a1 = a - a.int().float()

ind = th.LongTensor([0,2])

b = a[ind]

c = b * 5

c.backward([th.FloatTensor([5,5])])

print a.grad.data.numpy()


