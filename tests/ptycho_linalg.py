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
#from matplotlib.backends.backend_gtkagg import NavigationToolbar2GTKAgg as NaviToolbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse import csr_matrix
import scipy.sparse.linalg as linalg
from scipy.sparse.linalg.eigen.arpack import ArpackNoConvergence, ArpackError
import scipy.sparse as sp
import sys
import psutil
import gc as gc
#from joblib import Parallel, delayed
from numpy import linalg as LA

# Print iterations progress
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '█' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()
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
        

# returns a row of the 2d dft matrix with size N^2
class DFT2d():
    def __init__(self,N):
        i, j = np.ogrid[0:N,0:N]
        omega = np.exp( - 2 * pi * 1J / N )
    #        W2 = np.power( omega, i * j ) / sqrt(N)
        self.W2 = np.power( omega, i * j ) / sqrt(N)
        self.N = N

    def row(self,row):
#        applot(self.W2,'self.W2')
        self.rowrep = np.broadcast_to(self.W2[row % self.N],(self.N,self.N))
#        applot(self.rowrep,'self.rowrep')
#        applot(self.W2[row/self.N].reshape((self.N,1)),'self.W2[row/self.N].reshape((self.N,1))')        
        self.res = self.rowrep * self.W2[row/self.N].reshape((self.N,1))
#        applot(self.res,'self.res')
        return self.res.flatten()
        
    def row_mul_diag(self,row, diag):
        self.rowrep = np.broadcast_to(self.W2[row % self.N],(self.N,self.N))
        self.res = self.rowrep * self.W2[row/self.N].reshape((self.N,1))
        self.res = self.res.flatten() * diag
        return self.res

# return the result of left multiplication of the 2d dft matrix with a diagonal masking matrix
def DFT2d_leftmasked(mask):
    N = int(sqrt(mask.size))
    s = mask.sum() 
    print s
    print s*mask.size
    row = np.ndarray((s*mask.size,),dtype=np.int)
    col = np.ndarray((s*mask.size,),dtype=np.int)
    data = np.ndarray((s*mask.size,),dtype=np.complex64)
    dft = DFT2d(N)
    ones = np.ones(N**2).astype(np.int)
    cols = np.arange(N**2).astype(np.int)
    col[:] = np.tile(cols,s)
    i = 0
    for r,m in enumerate(mask):
        if m == 1:
#            print psutil.virtual_memory().used
            row[i*N**2:(i+1)*N**2] = ones * r
#            print psutil.virtual_memory().used
            data[i*N**2:(i+1)*N**2] = dft.row(r)
#            print psutil.virtual_memory().used
            i += 1
#            gc.collect()
            printProgress(i,s)            
#            print r
    print psutil.virtual_memory().used
    return csr_matrix((data, (row, col)), shape=(N**2,N**2))
    
def DFT2d_leftmasked_mul_diag(mask,diag):
    N = int(sqrt(mask.size))
    s = mask.sum() 
    print s
    print s*mask.size
    row = np.ndarray((s*mask.size,),dtype=np.int)
    col = np.ndarray((s*mask.size,),dtype=np.int)
    data = np.ndarray((s*mask.size,),dtype=np.complex64)

    dft = DFT2d(N)

    ones = np.ones(N**2).astype(np.int)
    cols = np.arange(N**2).astype(np.int)
    col[:] = np.tile(cols,s)
    i = 0
    for r,m in enumerate(mask):
        if m == 1:
#            print psutil.virtual_memory().used
            row[i*N**2:(i+1)*N**2] = ones * r
#            print psutil.virtual_memory().used
            data[i*N**2:(i+1)*N**2] = dft.row_mul_diag(r,diag)
#            print psutil.virtual_memory().used
            i += 1
#            gc.collect()
            printProgress(i,s)            
#            print r
    print psutil.virtual_memory().used
    return csr_matrix((data, (row, col)), shape=(N**2,N**2))
    
def largest_evec_Ta_Fv(Ta,v):
    N = int(sqrt(Ta.size))
    assert v.size == N**2
    TaF = DFT2d_leftmasked(Ta)
    applot(TaF.toarray(),'TaF')
    print psutil.virtual_memory().used
    TaFv = TaF.dot(sp.diags(v,0,(N**2,N**2)))
    print psutil.virtual_memory().used
    riplot(TaFv.toarray(),'TaFv')
    try:
        evals, evec = linalg.eigs(TaFv.toarray(),1, which='LM')
#        evals, evec = LA.eig(TaFv.toarray())
    except ArpackNoConvergence as anc:
        print anc.message
        print 'Eigenvector search did not converge'
    except ArpackError as ae:
        print ae.message
        print 'Arpack Error'
    print psutil.virtual_memory().used
    print evals
    applot(evec[:,0].reshape((N,N)),'evec')
    return evals.real, evals.imag, np.abs(evec), np.angle(evec)
#    return 1,2,3,4
    
def largest_evec_Ta_Fv2(Ta,v):
    N = int(sqrt(Ta.size))
    assert v.size == N**2
    TaFv = DFT2d_leftmasked_mul_diag(Ta,v)
    print psutil.virtual_memory().used
    riplot(TaFv.toarray(),'TaFv')
    try:
        evals, evec = linalg.eigs(TaFv,1, which='LM')
#        evals, evec = LA.eig(TaFv.toarray())
    except ArpackNoConvergence:
        print 'Eigenvector search did not converge'
    except ArpackError:
        print 'Arpack Error'
    print psutil.virtual_memory().used
    print evals
    applot(evec[:,0].reshape((N,N)),'evec')
    return evals.real, evals.imag, np.abs(evec), np.angle(evec)   

def largest_evec_Ta_Fv_reals(Ta,vreal,vimag):
    v = vreal + 1j* vimag
    return largest_evec_Ta_Fv(Ta,v)

def largest_evec_Ta_Fv_reals2(Ta,vreal,vimag):
    v = vreal + 1j* vimag
    return largest_evec_Ta_Fv2(Ta,v)
    
W = fft(np.eye(N))/ sqrt(N)
i, j = np.ogrid[0:N,0:N]
omega = np.exp( - 2 * pi * 1J / N )
W2 = np.power( omega, i * j ) / sqrt(N)
W2D = np.kron(W,W) 

#W2D_rowwise = np.ones((N**2,N**2)).astype(np.complex64)
#for i in range(N**2):
#    row = DFT2d_row(N,i)
#    W2D_rowwise[i] = row
    
#W2D_sparse = DFT2d_leftmasked(np.ones(N**2))
#zplot([np.angle(W2D_sparse.toarray()),np.angle(W2D)],'W2D sparse')
#zplot([np.angle(W2D_rowwise),np.angle(W2D)],'W2D row')


#sig = np.random.randn(N,N)
#sigflat = sig.ravel()

m = np.random.binomial(1,0.2,N**2)

v = np.random.random_integers(-10,10,N**2) + 1j* np.random.random_integers(-10,10,N**2)
print v

#evalre, evalim, evec_abs, evec_phase = largest_evec_Ta_Fv_reals(m,v.real,v.imag)
evalre, evalim, evec_abs, evec_phase = largest_evec_Ta_Fv_reals(m,v.real,v.imag)
print evalre[0], evalim[0]



#md = np.diag(m)
##print m.shape
#mW2 = md.dot(W2D)
##print mW2.shape
#mfft1 = mW2.dot(sigflat).reshape((N,N))/sqrt(N*N)
#
#sigfft1 = W2D.dot(sigflat).reshape((N,N))/sqrt(N*N)
#sigfft2 = np.fft.fft2(sig)/sqrt(N*N)
#
#mfft2 = np.reshape(m,(N,N))*sigfft2
#
##print np.allclose(W2D_rowwise,W2D)
#print np.allclose(W2D_sparse.toarray(),W2D)
