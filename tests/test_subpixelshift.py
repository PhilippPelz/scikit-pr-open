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
import skpr.core.parameters as p
import skpr.nn as n

par = p.get_ptycho_default_parameters()
io.init_logging(par)

import time
N = 256
b = 10
a = np.zeros((1,1,1,N,N)).astype(np.float32)
#print a.shape
a[:] = 0

E = 300e3
dp_size_out = 256.0
defocus_nm = 800
save_name = 'scan7_subframes_%d.h5' % defocus_nm
det_pix = 14 * 5e-6
alpha_rad = 4.6e-3
dx_angstrom = 1.32
# N = 1536
rs_rad = 0.1
fs_rad = 0.3
# r,i = blr_probe2(N,rs_rad,fs_rad,0.00,False)
r, i, fr, fi = focused_probe(E, N, d=dx_angstrom, alpha_rad=alpha_rad, defocus_nm=defocus_nm, det_pix=det_pix, C3_um=0, C5_mm=0, \
                     tx=0, ty=0, Nedge=5, plot=False)
pr = (r + 1j * i)

a[0, 0, 0, :, :] = pr
# a[0,0,0,10:20,10:20] = 1

io.plot(a[0,0,0])
epoch = th.FloatTensor([1])
subpixel_optimization_active=lambda epoch: True

shifter = M.SubpixelShift(a.shape, epoch, subpixel_optimization_active)

s = np.ones((1,2)).astype(np.float32)
s[0,0] = 0.0
s[0,1] = 10

ac = Variable(th.from_numpy(a + 0j).cuda(), requires_grad=True)
sc = th.from_numpy(s).cuda()

out = shifter(ac,sc)
o = out.data.cpu().numpy().real
io.plot(o[0,0,0], 'after')
io.plot(a[0,0,0]-o[0,0,0], 'before - after')

ground_truth = out.clone()

s1 = np.ones((1,2)).astype(np.float32)
s1[0,0] = 0.0
s1[0, 1] = 15.0
sc1 = Variable(th.from_numpy(s1).cuda(), requires_grad=True)

param = u.Param()
param.lr = 1
#param.max_eval = 5
param.nesterov = False
param.momentum = 0
#cg(opfunc, x, config, state=None):
config = {}
    
opt = th.optim.SGD([sc1], **param)

for i in range(100):
    opt.zero_grad()    
    
    def closure():
    
        out = shifter(ac,sc1)
    
#    o = out.data.cpu().numpy().real
#    io.plot(o[0,0,0], 'out')
#    o = ground_truth.data.cpu().numpy().real
#    io.plot(o[0,0,0], 'ground_truth')

        diff = ground_truth - out
#    o = diff.data.cpu().numpy().real
#    io.plot(o[0,0,0], 'diff')
    
        loss = (diff.data.norm() ** 2).real
    
        print 'loss :', loss 

        out.backward(diff)

        x = n.var['grad_shifts_all']
        io.plot_re_im_mosaic(x[0], 'grad_shifts_all 0')
        io.plot_re_im_mosaic(x[1], 'grad_shifts_all 1')
        
        return Variable(th.FloatTensor([loss]))
    def closure2(sh):
        sc1.data.copy_(sh)
        out = shifter(ac,sc1)
    
#    o = out.data.cpu().numpy().real
#    io.plot(o[0,0,0], 'out')
#    o = ground_truth.data.cpu().numpy().real
#    io.plot(o[0,0,0], 'ground_truth')

        diff = ground_truth - out
#    o = diff.data.cpu().numpy().real
#    io.plot(o[0,0,0], 'diff')
    
        loss = (diff.data.norm() ** 2).real
    
        print 'loss :', loss 

        out.backward(diff)
        
        return loss, sc1.grad.data
#    grad_shifts_all_x = fftshift(p.var['grad_shifts_all'][0,0,0,0]) # dimension: 2, K, N_o, N_p, M1, M2
#    grad_shifts_all_y = fftshift(p.var['grad_shifts_all'][1,0,0,0])
#    grad_ramp = fftshift(p.var['grad_ramp'][0,0,0]) # dimension: K, N_o, N_p, M1, M2
#    io.zplot([np.abs(grad_ramp),np.angle(grad_ramp)],'grad_ramp')
#    io.zplot([np.real(grad_shifts_all_x),np.imag(grad_shifts_all_x)],'grad_shifts_all_x', cmap=['hot', 'hot'])
#    io.zplot([np.real(grad_shifts_all_y),np.imag(grad_shifts_all_y)],'grad_shifts_all_y', cmap=['hot', 'hot'])
    opt.step(closure)
#    cg(closure2, sc1.data, config)
    
    print 'shift :', sc1.data.cpu().numpy()
    print 'dshift :', sc1.grad.data.cpu().numpy()
    
    time.sleep(1)
    
    



#o = ac.grad.data.cpu().numpy().real
#io.plot(o[0,0,0])
#io.plot(a[0,0,0]-o[0,0,0])