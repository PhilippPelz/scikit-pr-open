from pyE17.io import h5read
import numpy as np
import torch as th
import imageio
from skpr._ext import skpr_thnn
from skpr.inout import *
denoise = skpr_thnn.denoise
complex2pair = skpr_thnn.CudaZFloatComplex2Pair
pair2complex = skpr_thnn.CudaZFloatPair2Complex

one = np.ones((10,10)).astype(np.float)

a = one + 2j*one

zplot([a.real,a.imag],True)

at = th.from_numpy(a.astype(np.complex64))
atc = at.cuda()
atc2 = th.cuda.FloatTensor(2,*atc.size())
print atc2.size()

complex2pair(atc,atc2)

atpair = atc2.cpu().numpy()

showPlot(atpair[0],True,'atpair[0]')
showPlot(atpair[1],True,'atpair[1]')

atc.fill_(0+0j)
a = atc.cpu().numpy()
zplot([a.real,a.imag],True, 'before fill')

print atc2.size(), atc.size()
pair2complex(atc2,atc)

a = atc.cpu().numpy()
zplot([a.real,a.imag],True, 'back to complex')