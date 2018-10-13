# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 20:20:37 2016

@author: philipp
"""
N = 8
from math import *
from matplotlib import pyplot as plt
import numpy as np
from numpy.fft import fft, fft2, fftshift
from numpy.linalg import eig
from matplotlib.backends.backend_gtkagg import NavigationToolbar2GTKAgg as NaviToolbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse import csr_matrix, diags
import scipy.sparse.linalg as linalg
import scipy.sparse as sp
import sys
import psutil
import gc as gc

#plt.style.use('ggplot')
def zplot(img, suptitle='Image', savePath=None, cmap=['hot','hot'], title=['Abs','Phase'], show=True):
    im1, im2 = img
    fig, (ax1,ax2) = plt.subplots(1,2)
    div1 = make_axes_locatable(ax1)
    div2 = make_axes_locatable(ax2)
    fig.suptitle(suptitle, fontsize=20)
    imax1 = ax1.imshow(im1, interpolation='nearest', cmap=plt.cm.get_cmap(cmap[0]))
    imax2 = ax2.imshow(im2, interpolation='nearest', cmap=plt.cm.get_cmap(cmap[1]))
    cax1 = div1.append_axes("right", size="10%", pad=0.05)
    cax2 = div2.append_axes("right", size="10%", pad=0.05)
    cbar1 = plt.colorbar(imax1, cax=cax1)
    cbar2 = plt.colorbar(imax2, cax=cax2)
    ax1.set_title(title[0])
    ax2.set_title(title[1])
    plt.tight_layout()
    if show:
        plt.show()
    if savePath is not None:
        # print 'saving'
        fig.savefig(savePath + '.png', dpi=300)
def applot(img, suptitle='Image', savePath=None, cmap=['hot','hsv'], title=['Abs','Phase'], show=True):
    im1, im2 = np.abs(img), np.angle(img)
    fig, (ax1,ax2) = plt.subplots(1,2)
    div1 = make_axes_locatable(ax1)
    div2 = make_axes_locatable(ax2)
    fig.suptitle(suptitle, fontsize=20)
    imax1 = ax1.imshow(im1, interpolation='nearest', cmap=plt.cm.get_cmap(cmap[0]))
    imax2 = ax2.imshow(im2, interpolation='nearest', cmap=plt.cm.get_cmap(cmap[1]))
    cax1 = div1.append_axes("right", size="10%", pad=0.05)
    cax2 = div2.append_axes("right", size="10%", pad=0.05)
    cbar1 = plt.colorbar(imax1, cax=cax1)
    cbar2 = plt.colorbar(imax2, cax=cax2)
    ax1.set_title(title[0])
    ax2.set_title(title[1])
    plt.tight_layout()
    if show:
        plt.show()
    if savePath is not None:
        # print 'saving'
        fig.savefig(savePath + '.png', dpi=300)
def riplot(img, suptitle='Image', savePath=None, cmap=['hot','hot'], title=['Abs','Phase'], show=True):
    im1, im2 = np.real(img), np.imag(img)
    fig, (ax1,ax2) = plt.subplots(1,2)
    div1 = make_axes_locatable(ax1)
    div2 = make_axes_locatable(ax2)
    fig.suptitle(suptitle, fontsize=20)
    imax1 = ax1.imshow(im1, interpolation='nearest', cmap=plt.cm.get_cmap(cmap[0]))
    imax2 = ax2.imshow(im2, interpolation='nearest', cmap=plt.cm.get_cmap(cmap[1]))
    cax1 = div1.append_axes("right", size="10%", pad=0.05)
    cax2 = div2.append_axes("right", size="10%", pad=0.05)
    cbar1 = plt.colorbar(imax1, cax=cax1)
    cbar2 = plt.colorbar(imax2, cax=cax2)
    ax1.set_title(title[0])
    ax2.set_title(title[1])
    plt.tight_layout()
    if show:
        plt.show()
    if savePath is not None:
        # print 'saving'
        fig.savefig(savePath + '.png', dpi=300)        
#    plt.close()
def DFT2d_row(N,row):
    i, j = np.ogrid[0:N,0:N]
    omega = np.exp( - 2 * pi * 1J / N )
    W2 = np.power( omega, i * j ) / sqrt(N)
    rowrep = np.tile(W2[row % N],(N,1))
    res = rowrep * W2[row/N].reshape((N,1))  
    return res
    
# mask is length N^2
def DFT2d_leftmasked(mask,N):
    row = np.array([],dtype=np.int)
    col = np.array([],dtype=np.int)
    data = np.array([],dtype=np.complex64)
    
    cols = np.arange(N**2)
    for r,m in enumerate(mask):
        if m == 1:
            rows = np.ones(N**2) * r
            row = np.append(row, rows)
            col = np.append(col, cols)
            d = DFT2d_row(N,r).flatten()
            data = np.append(data,d)
#            print r
#    print row.shape
#    print col.shape
#    print data.shape
    return csr_matrix((data, (row, col)), shape=(N**2,N**2))

# Ta.size = N^2
# v.size = N^2    
def largest_evec_Ta_F_vec(Ta,v):
    N = int(sqrt(Ta.size))
    print 'before ev'
    Ta_F = DFT2d_leftmasked(Ta,N)
    print 'before ev'
    diagv = diags(v,0,(v.size,v.size))
    print 'before ev'
    Ta_Fv = Ta_F.dot(diagv)
#    riplot(Ta_Fv.toarray())
    # find the largest amplitude eigenvector and value
    print 'before ev'
    val, vec = linalg.eigs(Ta_Fv, k=1, which='LM')
    print 'After ev'
    print val
    vNN = vec.reshape((N,N))
    applot(vNN, 'vNN')
    
        
    
W = fft(np.eye(N))/ sqrt(N)
i, j = np.ogrid[0:N,0:N]
omega = np.exp( - 2 * pi * 1J / N )
W2 = np.power( omega, i * j ) / sqrt(N)


W2D = np.kron(W,W)
W2D_rowwise = np.ones((N**2,N**2)).astype(np.complex64)
for i in range(N**2):
    row = DFT2d_row(N,i)
    W2D_rowwise[i] = row.flatten()
    
W2D_sparse = DFT2d_leftmasked(np.ones(N**2),N)
#zplot([np.angle(W2D_sparse.toarray()),np.angle(W2D)],'W2D sparse')
#zplot([np.angle(W2D_rowwise),np.angle(W2D)],'W2D row')



sig = np.random.randn(N,N)
sigflat = sig.ravel()

m = np.random.binomial(1,0.5,N**2)
md = np.diag(m)

Ta = m

largest_evec_Ta_F_vec(Ta, np.ones(N**2))   


#print m.shape
mW2 = md.dot(W2D)
#print mW2.shape
mfft1 = mW2.dot(sigflat).reshape((N,N))/sqrt(N*N)

sigfft1 = W2D.dot(sigflat).reshape((N,N))/sqrt(N*N)
sigfft2 = np.fft.fft2(sig)/sqrt(N*N)

mfft2 = np.reshape(m,(N,N))*sigfft2


    

#eigval1, evec1= eig(mW2)
#eigval2, evec2= eig(np.diag(mfft2.ravel()))

#print(np.sort(np.abs(eigval1)))
#print(np.sort(np.abs(eigval2)))

#print 'sigfft shapes'
#print sigfft1.shape
#print sigfft2.shape
#f = plt.figure()
#ax = plt.imshow(np.angle(W))
#cb = plt.colorbar(ax)
#plt.show()
##
#f = plt.figure()
#ax = plt.imshow(np.abs(sigfft1))
#cb = plt.colorbar(ax)
#plt.show()
#
#f = plt.figure()
#ax = plt.imshow(np.abs(sigfft2))
#cb = plt.colorbar(ax)
#plt.show()
#
#f = plt.figure()
#ax = plt.imshow(np.abs(sigfft2))
#cb = plt.colorbar(ax)
#plt.show()

print np.allclose(W2D_rowwise,W2D)
print np.allclose(W2D_sparse.toarray(),W2D)
#print np.allclose(mfft1,mfft2)

#print rf.shape
#print qf.shape
import scipy.sparse.linalg as linalg
id = np.arange(20)
id[10] = 0
id[13] = 0
id[19] = 0
vals, vecs = linalg.eigsh(np.diag(id).astype(np.float32), k=3)
print vals
print vecs