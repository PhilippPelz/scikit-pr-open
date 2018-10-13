# -*- coding: utf-8 -*-

import numpy as np
import torch as th
from skpr.simulation.random_probe import * 
from skpr.inout.plot import *
from skpr.simulation.probe import focused_probe
from numpy.linalg import norm
from numpy.fft import fft2, fftshift, ifftshift
import math as m
def ortho(modes):
    """\
    Orthogonalize the given list of modes.
    """
    N = len(modes)
    A = np.array([[np.vdot(p2,p1) for p1 in modes] for p2 in modes])
    print 'A.shape ',A.shape
    e,v = np.linalg.eig(A)
    print 'e', e
    print 'v', v
    print v[:,0]
    ei = (-e).argsort()    
    print 'ei', ei
    nplist = [sum(modes[i] * v[i,j] for i in range(N)) for j in ei]
    amp = np.array([norm(npi)**2 for npi in nplist])
    amp /= amp.sum()
    return amp, nplist

def orthogonalize_modes(modes):
    """
    Orthogonalize the given tensor of modes.
    Parameters
    ----------
    modes : (Nmodes, Nx, Ny) tensor
        A complex tensor of shape (`Nx`, `Ny`) .
    Returns
    -------
    amp : (Nmodes) float tensor
        Relative intensities of the modes.
    nplist :  (Nmodes, Nx, Ny) tensor
        The orthogonal modes in descending order.
    """
    N = len(modes)
    A = np.array([[th.dot(p2.view(-1),p1.view(-1)) for p1 in modes] for p2 in modes])
    e,v = np.linalg.eig(A)
    v = th.from_numpy(v.astype(np.complex64))
    ei = (-e).argsort()  
    nplist = modes.clone()
    for k,j in enumerate(ei):
        new_mode = modes.clone()
        for i in range(N):
            new_mode[i] *= v[i,j]
            
        s = th.sum(new_mode, 0)
        print type(s), s.shape
        nplist[k].copy_(s)
    amp = th.FloatTensor([(nplist[i].norm()**2).real for i in range(N)])
    amp /= amp.sum()    
    return amp, nplist

N = 256
rs_rad = 0.1
fs_rad = 0.3
#r,i = blr_probe2(N,rs_rad,fs_rad,0.00,False)
r1,i1 = blr_probe2(N,rs_rad,fs_rad,0.00,False)

E = 300e3
dp_size_out = 256.0
N = 256
defocus_nm = 800
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


Nmodes= 3

p1 = r + 1j* i
p2 = np.roll(p1,5,0)
p3 = np.roll(p1,10,0)
#plotcx(p1)
#plotcx(p2)
print 'r.shape', r.shape

probes = np.zeros((Nmodes,r.shape[0],r.shape[1])).astype(np.complex64)
probes[0,...] = p1  
probes[1,...] = p2  
probes[2,...] = p3

a = probes.reshape((Nmodes,np.prod(r.shape)))
U, s, V = np.linalg.svd(a, full_matrices=False)

#ortho_svd = V.reshape((2,) + r.shape) 
#zplot([np.abs(m),np.angle(m)])

print 'svd sizes U,s,V = ', U.shape, s.shape,V.shape

amp, ortho = ortho(probes)
ortho = np.array(ortho)

amp1, ortho1 = orthogonalize_modes(th.from_numpy(probes).cuda())
ortho1 = ortho1.cpu().numpy()

print amp, amp1

#c = np.vdot(ortho_svd,ortho)
#phi = -np.angle(c)
#exp_minus_phi = np.cos(phi) + 1j* np.sin(phi)
#ortho = ortho * exp_minus_phi
    
#pc = th.from_numpy(probes.reshape((2,np.prod(r.shape))))
#print pc.size()
#print 'before pc.svd()'
#U1, s1, V1 = pc.svd()
#print 'after pc.svd()'

f = lambda x: ifftshift(fft2(fftshift(x)))

#print amp, amp2
print s**2/2

#for i in range(Nmodes):
#    ortho[i] /= m.sqrt(amp[i])
#    ortho1[i] /= m.sqrt(amp1[i])

print ortho.shape, ortho1.shape
plotcx(rect_mosaic(np.concatenate((ortho,ortho1))))
plot_probe_mosaic(th.from_numpy(np.concatenate((ortho,ortho1))).cuda())
#plotcx(mosaic(f(ortho)))

print np.vdot(ortho[0],ortho[1])
print np.vdot(f(ortho[0]),f(ortho[1]))