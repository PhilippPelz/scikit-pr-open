# -*- coding: utf-8 -*-

import numpy as np
from tifffile import TiffWriter
import matplotlib.pyplot as plt

def fermat_scan_roi(N_scan_x, N_scan_y, step):
    # simple code to create a fermat spiral scan
    n = np.arange(0, 1e4, 1)
    r = step * 0.57 * np.sqrt(n)
    theta = n * 137.508 / 180 * np.pi
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    ind_ok = np.logical_and(np.abs(x) < N_scan_x / 2, np.abs(y) < N_scan_y / 2)
    pos = np.zeros((ind_ok.sum(), 2)).astype(np.float32)
    pos[:, 0] = x[ind_ok]
    pos[:, 1] = y[ind_ok]
    return pos
def scatter_positions2(pos1, show=True, savePath=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(pos1[:, 0], pos1[:, 1], c='r', s=1)
    if savePath is not None:
        # print 'saving'
        fig.savefig(savePath + '.png', dpi=600)
    if show:
        plt.show()
def plot(img, title='Image', savePath=None, cmap='hot', show=True, vmax=None, figsize=(10, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(img, interpolation='nearest', cmap=plt.cm.get_cmap(cmap), vmax=vmax)
    cbar = fig.colorbar(cax)
    ax.set_title(title)
    plt.grid(False)
    if savePath is not None:
        fig.savefig(savePath + '.png', dpi=600)
    plt.show()
    fig.clf()

size = 512
dist = 8
savename = 'fermat_size_%03d_dist_%d' % (size, dist)

pos = fermat_scan_roi(size, size, dist)
pos -= pos.min(0)
print('Created %d positions' % pos.shape[0])
scatter_positions2(pos, show=True)
pos = pos.astype(np.int)
bin_mask = np.zeros((size, size))
for p in pos:
    bin_mask[p[0], p[1]] = 1

plot(bin_mask, title=savename, savePath=savename)

with TiffWriter(savename + '.tif') as tif:
    tif.save((bin_mask * 255).astype(np.int8))
