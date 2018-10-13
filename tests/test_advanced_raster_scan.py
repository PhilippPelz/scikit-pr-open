# -*- coding: utf-8 -*- 
import math as m 
import numpy as np 
from numpy.linalg import norm 
#from skpr.optim import HistorySGD 
import skpr.inout as io 
from skpr.core.parameters import * 
from skpr.core import get_ptycho_default_parameters 
import skpr.core.ptycho as pty 
from skpr.inout.h5rw import h5read 
from skpr.nn import modules as M 
from skpr.simulation.probe import focused_probe 
import skpr.util as u 
from numpy.fft import fftshift, ifftshift, fft2 
import matplotlib.pyplot as plt 
N = 5 
theta = -38
 
#pos = np.array(u.raster_scan(N,N,1,1)).astype(np.float32) 
pos = u.advanced_raster_scan(ny=N ,nx=N, fast_axis=1, mirror = [-1,1], theta=theta, dy=1, dx=1) 
 
print pos.shape 
for i in np.arange(1,N*N+1,1): 
    fig, ax = plt.subplots() 
    print pos[i-1] 
    ax.scatter(pos[:i, 1], pos[:i, 0], c='r') 
    plt.xlim(-N,N) 
    plt.ylim(-N,N) 
    plt.gca().invert_yaxis()
