# -*- coding: utf-8 -*-

import numpy as np
import torch as th
from skpr.simulation.random_probe import * 
from skpr.inout.plot import *
from skpr.simulation.probe import focused_probe
from numpy.linalg import norm
from numpy.fft import fft2, fftshift, ifftshift
from skpr.util import sector_mask

E = 300e3
dp_size_out = 256.0
N = 256
defocus_nm = 1200
save_name = 'scan7_subframes_%d.h5' % defocus_nm
det_pix = 14 * 5e-6
alpha_rad = 4.6e-3
dx_angstrom = 1.32
# N = 1536
rs_rad = 0.1
fs_rad = 0.3
# r,i = blr_probe2(N,rs_rad,fs_rad,0.00,False)
r, i = focused_probe(E, N, d=dx_angstrom, alpha_rad=alpha_rad, defocus_nm=defocus_nm, det_pix=det_pix, C3_um=0, C5_mm=0, \
                     tx=0, ty=0, Nedge=5, plot=False)
pr = r + 1j * i
plotcx(pr)
prt = th.from_numpy(pr).cuda()

from si_prefix import si_format
print si_format(1e-10) + 'A'

def determine_probe_radius(p):
    if p.ndimension() == 2:
        p = p.unsqueeze(0)
    s = p.shape
    is_cuda = 'cuda' in str(type(p))
    c = th.cuda if is_cuda else th
    I_P = c.FloatTensor(p.size())
    probe_intensity = p.expect(out=I_P)
    probe_intensity = th.sum(probe_intensity, 0, keepdim=False)
    center = np.array(probe_intensity.size())/2
    total_intensity = probe_intensity.sum()
    for radius in np.linspace(1,p.shape[1]/2,50):
        mask = th.from_numpy(sector_mask(probe_intensity.size(),center,radius,(0,360)).astype(np.float32))
        if is_cuda: mask = mask.cuda()
        radius_intensity = (probe_intensity * mask).sum()
#        print radius, radius_intensity/total_intensity
        if (radius_intensity/total_intensity) > 0.99:
            return radius
        
r = determine_probe_radius(prt)
print 
print r