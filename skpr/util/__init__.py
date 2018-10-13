import sys
from cmath import *

import h5py
import scipy.stats as stats
import torch as th
from numpy.fft import fftshift


from data_preparation import *
from parameters import *
from pattern import *
from physics import *




def get_shift_ramp(sh, neg=True):
    s = np.array(sh)
    x, y = np.mgrid[(-s[-2] / 2):(s[-2] / 2):1, -s[-1] / 2:s[-1] / 2:1]
    sh = (2,) + x.shape
    r = np.ones(sh).astype(np.float32)
    r[0] = x.astype(np.float32) / s[-2]
    r[1] = y.astype(np.float32) / s[-1]
    if not neg:
        r *= -1
    ramp = fftshift(r, axes=(1, 2))
    return ramp

def plot(img, title='Image', savePath=None, cmap='hot', show=True):
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(img, interpolation='nearest', cmap=plt.cm.get_cmap(cmap))
    cbar = fig.colorbar(cax)
    ax.set_title(title)
    plt.grid(False)
    #    fig.savefig(savePath + '.png', dpi=600)
    plt.show()
    fig.clf()

def rebin(a, s, mode='sum'):
    '''rebin ndarray data into a smaller ndarray of the same rank whose dimensions
    are factors of the original dimensions. eg. An array with 6 columns and 4 rows
    can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
    example usages:
    >>> a=rand(6,4); b=rebin(a,(3,2))
    >>> a=rand(6); b=rebin(a,2)
    '''
    shape = np.asarray(a.shape)
    lenShape = len(shape)
    args = np.ones_like(shape) * np.asarray(s)
    factor = shape // args
    s1 = tuple()
    for i in range(lenShape):
        s1 += (factor[i],)
        s1 += (args[i],)
    ret = a.reshape(s1)
    for i in range(lenShape):
        j = (lenShape - i) * 2 - 1
        ret = ret.sum(j)
    if mode == 'mean':
        ret /= np.prod(args)
    return ret


def plot(img, title='Image', savePath=None, cmap='hot', show=True):
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(img, interpolation='nearest', cmap=plt.cm.get_cmap(cmap))
    cbar = fig.colorbar(cax)
    ax.set_title(title)
    plt.grid(False)
    #    fig.savefig(savePath + '.png', dpi=600)
    plt.show()
    fig.clf()


#  local x = torch.repeatTensor(torch.linspace(-Nx/2,Nx/2,Nx),Ny,1)
#  -- pprint(x)
#  local y = x:clone():t()
#  local r = (x:pow(2) + y:pow(2)):sqrt()
#
#  local r_out = r:clone():add(-radius):abs():float()
#  -- plt:plot(r_out,'r_out')
#  r_out:div(-2*(out_scaling/2.35482)^2):exp()
#  -- plt:plot(r_out,'r_out')
#
#  local inside = torch.le(r, radius):cuda()
#  local outside1 = torch.gt(r, radius):float():cmul(r_out)
#  local outside = outside1:cuda()
#
#  self.mask = inside:add(outside)
#
#  if fft_shift then
#    self.mask:fftshift()
#  end

def determine_probe_radius(p):
    if p.ndimension() == 2:
        p = p.unsqueeze(0)
    s = p.shape
    is_cuda = 'cuda' in str(type(p))
    c = th.cuda if is_cuda else th
    I_P = c.FloatTensor(p.size())
    probe_intensity = p.expect(out=I_P)
    probe_intensity = th.sum(probe_intensity, 0, keepdim=False)
    # plot(probe_intensity.cpu().numpy())
    center = np.array(probe_intensity.size())/2
    total_intensity = th.sum(th.sum(probe_intensity, 0, keepdim=False), 0, keepdim=False)[0]
    # print 'total_intensity ', total_intensity, probe_intensity.cpu().numpy().sum()
    for radius in np.linspace(1, p.shape[1] / 2, 30):
        mask = th.from_numpy(sector_mask(probe_intensity.size(),center,radius,(0,360)).astype(np.float32))
        if is_cuda: mask = mask.cuda()
        # plot((probe_intensity * mask).cpu().numpy())
        radius_intensity = (probe_intensity * mask).sum()
#        print radius, radius_intensity/total_intensity
        if (radius_intensity/total_intensity) > 0.97:
            return radius
    return p.shape[1]/2


def print_hdf5_item_structure(g, offset='    '):
    """Prints the input file/group/dataset (g) name and begin iterations on its content"""
    if isinstance(g, h5py.File):
        print g.file, '(File)', g.name

    elif isinstance(g, h5py.Dataset):
        print '(Dataset)', g.name, '    len =', g.shape  # , g.dtype

    elif isinstance(g, h5py.Group):
        print '(Group)', g.name

    else:
        print 'WORNING: UNKNOWN ITEM IN HDF5 FILE', g.name
        sys.exit("EXECUTION IS TERMINATED")

    if isinstance(g, h5py.File) or isinstance(g, h5py.Group):
        for key, val in dict(g).iteritems():
            subg = val
            print offset, key,  # ,"   ", subg.name #, val, subg.len(), type(subg),
            print_hdf5_item_structure(subg, offset + '    ')

def memory_report():
    print('=== Memory report ========================')
    total = 0
    for obj in gc.get_objects():
        if th.is_tensor(obj):
            s = 2 ** (np.log2(obj.nelement() * 4) - 20)
            total += s
            print('%s %f MB' % (type(obj), s))
        if (hasattr(obj, 'data') and th.is_tensor(obj.data)):
            s = 2 ** (np.log2(obj.data.nelement() * 4) - 20)
            total += s
            print('%30s %f MB' % (type(obj), s))
    print('------------------------------------------')
    print('total: %f MB' % total)
    print('==========================================')

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
        Relative amplitudes of the modes.
    nplist :  (Nmodes, Nx, Ny) tensor
        The orthogonal modes in descending order.
    """
    N = len(modes)
    A = np.array([[th.dot(p2.view(-1), p1.view(-1)) for p1 in modes] for p2 in modes])
    e, v = np.linalg.eig(A)
    v = th.from_numpy(v.astype(np.complex64))
    ei = (-e).argsort()
    nplist = modes.clone()
    for k, j in enumerate(ei):
        new_mode = modes.clone()
        for i in range(N):
            new_mode[i] *= v[i, j]

        s = th.sum(new_mode, 0)
        # print type(s), s.shape
        nplist[k].copy_(s)
    amp = th.FloatTensor([(nplist[i].norm() ** 2).real for i in range(N)])
    amp /= amp.sum()
    return amp, nplist


def truncnorm(shape, a, b, mu, sigma):
    T = stats.truncnorm(
        (a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma)
    return T.rvs(shape)


def exp_decay_mask(shape, radius, factor, shift=False):
    x, y = np.ogrid[:shape[0], :shape[1]]
    cx, cy = (shape[0] / 2, shape[1] / 2)
    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    r = np.sqrt(r2)
    inside = r <= radius
    outside = r > radius
    r_out = np.exp(np.abs(r - radius) / (-2 * (factor / 2.35482) ** 2))
    mask = inside + (outside * r_out)
    if shift:
        mask = fftshift(mask)

    return mask.astype(np.float32)


def dist(z, x):
    #    print z.view(-1).size()
    #    print x.view(-1).size()
    c = z.view(-1).dot(x.view(-1))
    phi = -phase(c)
    exp_minus_phi = cos(phi) + 1j * sin(phi)
    x_hat = x * exp_minus_phi
    # plotReIm(x_hat.cpu().numpy(),'x_hat')
    # plotReIm(z.cpu().numpy(), 'z')
    diff = z - x_hat
    dist = diff.norm().real
    return dist


def rel_dist(z, x):
    d = dist(z, x)
    x_norm = x.view(-1).norm().real
    return d / x_norm


def sector_mask(shape, centre, radius, angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in
    `angle_range` should be given in clockwise order.
    """

    x, y = np.ogrid[:shape[0], :shape[1]]
    cx, cy = centre
    tmin, tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
        tmax += 2 * np.pi

    # convert cartesian --> polar coordinates
    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    theta = np.arctan2(x - cx, y - cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2 * np.pi)

    # circular mask
    circmask = r2 <= radius * radius

    # angular mask
    anglemask = theta <= (tmax - tmin)

    return circmask * anglemask

    # x = exp_decay_mask((128,128),30,6,False)
    # plot(x,'mask')
