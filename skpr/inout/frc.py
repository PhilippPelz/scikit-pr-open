from pyE17.utils import imsave
from matplotlib import pyplot as plt
import numpy as np
from numpy.fft import fft2, ifftshift, fftshift, ifft2, fft, ifft
from scipy.ndimage.filters import gaussian_filter1d
from .plot import imsave

def taperarray(shape, edge):
    xx, yy = np.mgrid[0:shape[0], 0:shape[1]]
    xx1 = np.flipud(xx)
    xx2 = np.minimum(xx, xx1)
    yy1 = np.fliplr(yy)
    yy2 = np.minimum(yy, yy1)
    rr = np.minimum(xx2, yy2).astype(np.float)

    rr[rr <= edge] /= edge
    rr[rr > edge] = 1
    rr *= np.pi / 2
    rr = np.sin(rr)
    #    print xx2
    #    fig, ax = plt.subplots()
    #    imax = ax.imshow(rr)
    #    plt.colorbar(imax)
    #    plt.show()
    return rr


def tapercircle(shape, edge, edgeloc=8 / 20.0, loc=(0, 0)):
    x0, y0 = loc
    xx, yy = np.meshgrid(np.arange(-shape[0] / 2, shape[0] / 2), np.arange(-shape[1] / 2, shape[1] / 2))
    rr = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)
    rr -= (shape[0] * edgeloc - edge)
    one = rr < 0
    taper = np.logical_and(rr >= 0, rr <= edge)
    zero = rr > edge

    #    rr[taper]
    rr[zero] = 0
    rr[taper] = np.abs(rr[taper] - np.max(rr[taper])) / np.max(rr[taper])

    rr[one] = 1

    return rr

def frc(im1, im2, annulus_width=1, edgetaper=5, edgeloc=8 / 20.0, loc=(0, 0), smooth=None, working_mask=None, x=None,
        y=None, rmax=None, taper=None):
    """
    r = radial_data(data,annulus_width,working_mask,x,y)

    A function to reduce an image to a radial cross-section.

    :INPUT:
      data   - whatever data you are radially averaging.  Data is
              binned into a series of annuli of width 'annulus_width'
              pixels.

      annulus_width - width of each annulus.  Default is 1.

      working_mask - array of same size as 'data', with zeros at
                        whichever 'data' points you don't want included
                        in the radial data computations.

      x,y - coordinate system in which the data exists (used to set
               the center of the data).  By default, these are set to
               integer meshgrids

      rmax -- maximum radial value over which to compute statistics

    :OUTPUT:
        r - a data structure containing the following
                   statistics, computed across each annulus:

          .r      - the radial coordinate used (outer edge of annulus)

          .mean   - mean of the data in the annulus

          .sum    - the sum of all enclosed values at the given radius

          .std    - standard deviation of the data in the annulus

          .median - median value in the annulus

          .max    - maximum value in the annulus

          .min    - minimum value in the annulus

          .numel  - number of elements in the annulus

    :EXAMPLE:
      ::

        import numpy as np
        import pylab as py
        import radial_data as rad

        # Create coordinate grid
        npix = 50.
        x = np.arange(npix) - npix/2.
        xx, yy = np.meshgrid(x, x)
        r = np.sqrt(xx**2 + yy**2)
        fake_psf = np.exp(-(r/5.)**2)
        noise = 0.1 * np.random.normal(0, 1, r.size).reshape(r.shape)
        simulation = fake_psf + noise

        rad_stats = rad.radial_data(simulation, x=xx, y=yy)

        py.figure()
        py.plot(rad_stats.r, rad_stats.mean / rad_stats.std)
        py.xlabel('Radial coordinate')
        py.ylabel('Signal to Noise')
    """

    # 2012-02-25 20:40 IJMC: Empty bins now have numel=0, not nan.
    # 2012-02-04 17:41 IJMC: Added "SUM" flag
    # 2010-11-19 16:36 IJC: Updated documentation for Sphinx
    # 2010-03-10 19:22 IJC: Ported to python from Matlab
    # 2005/12/19 Added 'working_region' option (IJC)
    # 2005/12/15 Switched order of outputs (IJC)
    # 2005/12/12 IJC: Removed decifact, changed name, wrote comments.
    # 2005/11/04 by Ian Crossfield at the Jet Propulsion Laboratory

    import numpy as ny

    class radialDat:
        """Empty object container.
        """

        def __init__(self):
            self.num = None
            self.denom = None
            self.T1bit = None
            self.Thalfbit = None

    # ---------------------
    # Set up input parameters
    # ---------------------

    if working_mask == None:
        working_mask = ny.ones(im1.shape, bool)

    npix, npiy = im1.shape
    if taper is not None:
        taper0 = taper
    else:
        taper0 = tapercircle(im1.shape, edgetaper, edgeloc, loc)
    f, a = plt.subplots()
    a.imshow(imsave(im1 * taper0))
    plt.title('tapered')
    plt.show()
    #    taper0 = 1
    F1 = fftshift(fft2(ifftshift(im1 * taper0)))
    F2 = fftshift(fft2(ifftshift(im2 * taper0)))
    F1F2_star = F1 * F2.conj()

    if x == None or y == None:
        x1 = ny.arange(-npix / 2., npix / 2.)
        y1 = ny.arange(-npiy / 2., npiy / 2.)
        x, y = ny.meshgrid(y1, x1)

    r = abs(x + 1j * y)

    if rmax == None:
        rmax = r[working_mask].max()

    # ---------------------
    # Prepare the data container
    # ---------------------
    dr = ny.abs([x[0, 0] - x[0, 1]]) * annulus_width
    radial = ny.arange(rmax / dr) * dr + dr / 2.
    nrad = len(radial)
    radialdata = radialDat()
    radialdata.num = ny.zeros(nrad)
    radialdata.denom = ny.zeros(nrad)
    radialdata.T1bit = ny.zeros(nrad)
    radialdata.Thalfbit = ny.zeros(nrad)
    radialdata.r = radial / (npix / 2)

    # ---------------------
    # Loop through the bins
    # ---------------------
    for irad in range(nrad):  # = 1:numel(radial)
        minrad = irad * dr
        maxrad = minrad + dr
        thisindex = (r >= minrad) * (r < maxrad) * working_mask
        # import pylab as py
        # pdb.set_trace()
        if not thisindex.ravel().any():
            radialdata.num[irad] = ny.nan
            radialdata.denom[irad] = ny.nan
            radialdata.T1bit[irad] = ny.nan
            radialdata.Thalfbit[irad] = ny.nan
        else:
            sqrt_n = np.sqrt(thisindex.astype(np.int).sum())
            radialdata.num[irad] = np.real(F1F2_star[thisindex].sum())
            radialdata.denom[irad] = ny.sqrt((ny.abs(F1[thisindex]) ** 2).sum() * (ny.abs(F2[thisindex]) ** 2).sum())
            radialdata.T1bit[irad] = (0.5 + 2.4142 / sqrt_n) / (1.5 + 1.4142 / sqrt_n)
            radialdata.Thalfbit[irad] = (0.2071 + 1.9102 / sqrt_n) / (1.2071 + 0.9102 / sqrt_n)

    # ---------------------
    # Return with data
    # ---------------------
    radialdata.frc = ny.nan_to_num(radialdata.num / radialdata.denom)
    radialdata.frc[radialdata.frc < 0] = 0
    if smooth is not None:
        radialdata.frc = gaussian_filter1d(radialdata.frc, smooth)
    take = radialdata.r <= 1.1
    radialdata.r = radialdata.r[take]
    radialdata.frc = radialdata.frc[take]
    radialdata.T1bit = radialdata.T1bit[take]
    radialdata.Thalfbit = radialdata.Thalfbit[take]
    return radialdata