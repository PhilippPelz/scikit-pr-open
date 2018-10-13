#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 10:11:21 2017

@author: philipp
"""
from numpy.compat import integer_types
from matplotlib import pyplot as plt
from skpr.inout.h5rw import h5read, h5write
from skpr.inout import viewing as v
import skpr.inout as io
from skpr.nn.modules import *
from numpy.fft import fftshift, fft2
from skpr.core.engines import BaseEngine
from skpr.util import *
import skpr.util as u
from skpr.core import get_ptycho_default_parameters
from skpr.core.ptycho.models import FarfieldPtychographyNet
from skpr.inout.plot import plot, zplot, plotmosaic
from skpr.nn import modules as M

from skpr.simulation.probe import focused_probe

from skimage import data
from scipy.ndimage.interpolation import rotate

oa = data.camera().astype(np.float32)
oa /= oa.max()
oa *= 0.2
oa += 0.8
oph = rotate(oa, 180)
ob = oa * np.exp(1j * 2 * np.pi * oph)
x = 20
ob = ob[320-x:510-x,185-x:375-x]
io.zplot([np.abs(ob), np.angle(ob)])

f = h5read('/home/philipp/dropbox/Public/moon5.h5')
# for k in f:
#     print k
# pr = f['pr'] + 1j*f['pi']
# pr = pr[0][0]
E = 300e3
dp_size_out = 256.0
N = 128
defocus_nm = 0
save_name = 'scan7_subframes_%d.h5' % defocus_nm
det_pix = 14 * 5e-6
alpha_rad = 4.6e-3
dx_angstrom = 1.32
# N = 1536
rs_rad = 0.13
fs_rad = 0.3
# r, i = blr_probe2(N, rs_rad, fs_rad, 0.00, False)
r,i = focused_probe(E, N, d = dx_angstrom, alpha_rad=alpha_rad, defocus_nm = defocus_nm, det_pix = det_pix, C3_um = 0, C5_mm=0, \
                     tx = 0,ty =0, Nedge = 0, plot=True)
pr = r + 1j * i
# r,i = fzp(128,300)
sx = N * (rs_rad + 0.02)
pr = r + 1j * i
pr *= 1e4
c = np.array(pr.shape) / 2
print c
s = 256
# pr = pr[c[0]-s:c[0]+s,c[1]-s:c[1]+s] * 1e6


# ob = f['or'] + 1j*f['oi']
# ob = ob[0][0]
sh = ob.shape
# o1 = np.ones((sh[0]+N/2,sh[1]+N/2)).astype(np.complex64)
# o1[N/2:,N/2:] = ob
# ob = o1
io.zplot([np.abs(pr), np.angle(pr)], 'probe')
d = th.from_numpy(f['data_unshift'])

p = get_ptycho_default_parameters()

p.probe.initial = th.from_numpy(pr)
p.model = FarfieldPtychographyNet

# u = th.from_numpy(np.random.randn(10,100,100) + 1j*  np.random.randn(10,100,100)).cuda()
# print u.norm(), u.fft2().norm()
# pos = f['scan_info']['positions']
# po = np.array(u.spiral_scan_roi(10, sh[0] - N / 2, sh[1] - N / 2))
po = np.array(u.raster_scan(ny=32,nx=32,dy=1,dx=1))

pos = th.from_numpy(po.astype(np.float32))
minpos, minind = pos.min(0)
print minpos
p.ptycho.pos = pos.add(-1, minpos.expand_as(pos)).int().float()
print p.ptycho.pos
minpos, minind = p.ptycho.pos.min(0)
print minpos

p.y = th.from_numpy(np.random.randn(100, N, N).astype(np.float32))
p.valid_mask = th.zeros_like(p.y).float()
print 'obshape', ob.shape
p.object.initial = th.from_numpy(ob)
p.object.margins = 10
p.loss.function = M.TruncatedFarFieldPoissonLikelihood
# print p.ptycho.pos.float()

p.optimizer.type = th.optim.SGD

eng = BaseEngine(p)
I = eng.model_output()

I = I.data.cpu().numpy().astype(np.float32)
print I.shape
# print I.size()
##v.show3d(d)

plot(np.log10(I[0]))
plot(np.log10(I[1]))
plotmosaic(fftshift(I, axes=(1, 2)))

print type(pr)
print type(ob)
print type(I)
print type(p.ptycho.pos)

h5write('/home/philipp/projects/simWDD.h5', pr=pr, preal=pr.real, pimag=pr.imag, ob=ob, oreal=ob.real, oimag=ob.imag, I=I,
        pos=p.ptycho.pos.numpy())
#
# a = np.array([1+1j,2+2j])
# b = np.array([3+3j,4+4j])
# ath = th.from_numpy(a)
# bth = th.from_numpy(b)
# ath_cuda = ath.cuda()
# ath_cuda += bth.cuda()
# print(ath_cuda.storage().data_ptr())
# ath_cuda = bth.cuda().abs()
# print(ath_cuda.storage().data_ptr())
# ath = ath_cuda.cpu()
# print(ath.numpy())
#
# x = sector_mask((128,128),(64,64),30,(0,360))
#
# cx = (x+0j).astype(np.complex64)
# cxth = th.from_numpy(cx)
# c1 = cxth.cuda()
# print(type(c1))
# c2 = c1.clone()
#
# d = th.FloatTensor(c1.size())
# print d.size()
# s = d.size()
# c3 = c2.view(th.Size([1,s[0],s[1]]))
# print c3.size()
# c3 = c3.expand(5,s[0],s[1])
# c3 = c3.contiguous()
# c4 = c3.clone()
# print c3.size()
# c2.fft2(out=c1)
# c3.fft2(out=c4)
# print c4.size()
# print 'norm', c1.norm()
#
# c1.re(out=c2)
#
# f,a = plt.subplots()
# a.imshow(np.log10(np.abs(c2.cpu().numpy())))
# plt.plot()
#
# c2 = c2.fftshift()
# a1 = c2.cpu().numpy()
#
# f,a = plt.subplots()
# a.imshow(np.log10(np.abs(c2.cpu().numpy())))
# plt.plot()
#
# c2.abs_()

