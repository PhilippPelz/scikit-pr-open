#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:45:00 2018

@author: pelzphil
"""

from __future__ import print_function

import gc
import json
import math as m

import h5py
import mrcfile
import numpy as np
import progressbar
import pytiff
import skpr.inout as io
import skpr.util as u
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from numpy.linalg import norm
from si_prefix import si_format
from skpr.simulation import probe


class DataPreparation:
    def __init__(self, path='./', name='R4_00009', mask_file='./Ref12bit_pxmask.tif', \
                 reference_file='./Ref12bit_reference.mrc', q_max_rel=1.1, exclude_indices=[], \
                 binning_factor=1, min_fraction_valid=0.5, interpolate_dead_pixels=True, binary_mask_file=None,
                 save_suffix='processed', fast_axis=1, save_hdf5=True, save_matlab=False):
        self.path = path
        self.name = name
        self.mask_file = mask_file
        self.reference_file = reference_file
        self.q_max_rel = q_max_rel
        self.exclude_indices = exclude_indices
        self.binning_factor = binning_factor
        self.min_fraction_valid = min_fraction_valid
        self.interpolate_dead_pixels = interpolate_dead_pixels
        self.binary_mask_file = binary_mask_file
        self.save_suffix = save_suffix
        self.fast_axis = fast_axis
        self.save_hdf5 = save_hdf5
        self.save_matlab = save_matlab

        self.radius = np.array([250])
        self.radius_aperture = np.array([250])
        self.c1 = np.array([0, 0, 0])
        self.c2 = np.array([0, 0, 0])
        self.pos1 = np.array([0, 0, 0])
        self.pos2 = np.array([0, 0, 0])

        self.valid_mask = None
        self.bin_mask_positions = None
        self.data = None
        self.pos = None

    def gap_mask(self, ms):
        gm = np.ones(ms, dtype=np.float32)
        b = 3
        gm[ms[0] / 2 - b:ms[0] / 2 + b, :] = 0
        gm[:, ms[1] / 2 - b:ms[1] / 2 + b] = 0
        gm[:, ms[1] / 2 - 256 - b - b:ms[1] / 2 - 256 - b + b] = 0
        gm[:, ms[1] / 2 + 256 + b - b:ms[1] / 2 + 256 + b + b] = 0
        gm[:, ms[1] / 2 - 512 - 3 * b - b:ms[1] / 2 - 512 - 3 * b + b] = 0
        gm[:, ms[1] / 2 + 512 + 3 * b - b:ms[1] / 2 + 512 + 3 * b + b] = 0
        return gm

    def load_binary_position_mask(self):
        if self.binary_mask_file is not None:
            print('Loading binary position mask ...')
            with pytiff.Tiff(self.binary_mask_file) as handle:
                im = np.array(handle) * -1
                # io.plot(im,'positions')
                bin_mask_positions = im.astype(np.bool)
        else:
            bin_mask_positions = np.ones((self.stepy, self.stepx)).astype(np.bool)
        return bin_mask_positions

    def load_hot_pixel_mask(self, s):
        print('Loading hot pixel mask ...')
        if self.mask_file is not None:
            with pytiff.Tiff(self.mask_file) as handle:
                im = np.array(handle)
                mask = 1 - im / 255
        else:
            mask = np.ones((s[1], s[2]))
        return mask

    def load_correction_factor(self, s):
        print('Loading correction factors ...')
        if self.reference_file is not None:
            with mrcfile.open(self.reference_file) as mrc:
                correction_factor = 1 / mrc.data
        else:
            correction_factor = np.ones((s[1], s[2]))
        return correction_factor

    def prepare_stem_image(self):
        print('Preparing STEM image ...')
        s = self.data.shape

        if self.binary_mask_file is None:
            a1 = np.reshape(self.data, (self.stepy, self.stepx, s[1], s[2]))
            #    xs,ys = 11,20
            #    stepx = stepy = 10
            a2 = a1  # [xs:xs+stepx,ys:ys+stepy]
            a1 = None

            intensities = np.sum(a2, (2, 3))
            STEM = np.ones((self.stepy, self.stepx)).astype(np.float32) * np.mean(intensities)
            STEM.flat[:-1] = intensities.flat
            STEM.flat[-1] = np.mean(intensities.flat)

        else:
            intensities = np.sum(self.data, (1, 2))

            ind = np.linspace(0, s[0], endpoint=False, num=s[0]).astype(np.int)

            indices = np.zeros((self.stepy, self.stepx)).astype(np.int)
            indices[self.bin_mask_positions] = ind
            # io.plot(indices,'indices')

            for ind1 in self.exclude_indices:
                ind[ind1] = 0

            # [print(x,y) for x,y in zip(xpos,ypos)]
            # io.plot(bin_mask_positions, 'positions')
            STEM = np.zeros((self.stepy, self.stepx)).astype(np.float32)
            STEM[self.bin_mask_positions] = intensities
            from scipy.ndimage.filters import gaussian_filter
            STEM = gaussian_filter(STEM, sigma=10)

        io.plot(STEM, 'STEM', savePath='%s_STEM' % self.name)

    def determine_center_rotation_alpha(self):
        print('Determine center, alpha, and rotation angle ...')
        action_sequence = [
            ('Please match the radius of the diffraction disc CONTROL', 'control', 'r', self.radius),
            ('Please match the outer rim radius of the aperture CONTROL', 'control', 'r', self.radius_aperture),
            ('Now determine the center of the disc with your cursor ENTER', 'enter', 'pos', self.c1),
            (
                'Now move with the arrow keys to the diffraction pattern at the end of the line\n and determine the center again. ENTER',
                'enter', 'pos', self.c2),
            ('Now choose a feature in the disc and put cursor in the position. ENTER', 'enter', 'pos', self.pos1),
            ('Now move with the arrow keys and select the position of the same feature again. ENTER', 'enter', 'pos',
             self.pos2),
            ('Closing', 'close', 'pos', self.pos2)
        ]
        cursor = u.InteractiveDataPrep(self.data * self.valid_mask, 230, action_sequence)

    def crop_data(self):
        c = self.c
        M = self.M
        M2 = self.M2
        s = self.data.shape
        cropped_data = np.zeros((s[0], M, M), dtype=np.float32)
        ds = cropped_data.shape

        n_batches = 10
        batch_size = s[0] / n_batches

        print('Cropping data at %d,%d' % (c[0], c[1]))

        c0s = c[0] - M2 if (c[0] - M2) > 0 else 0
        c1s = c[1] - M2 if (c[1] - M2) > 0 else 0
        c0e = c[0] + M2 if (c[0] + M2) < s[1] else s[1]
        c1e = c[1] + M2 if (c[1] + M2) < s[2] else s[2]
        # print(c0s, c1s, c0e, c1e)

        c0size0 = M2 if (c[0] - M2) > 0 else c[0]
        c1size0 = M2 if (c[1] - M2) > 0 else c[1]
        c0size1 = M2 if (c[0] + M2) < s[1] else s[1] - c[0]
        c1size1 = M2 if (c[1] + M2) < s[2] else s[2] - c[1]
        # print(c0size0, c1size0, c0size1, c1size1)

        cropped_correction_factor = self.correction_factor[c0s:c0e, c1s:c1e]
        for i in range(n_batches + 1):
            print('Batch %0d ...' % i)
            start = i * batch_size
            end = (i + 1) * batch_size if (i + 1) * batch_size <= ds[0] else ds[0]

            crop = self.data[start:end, c0s:c0e, c1s:c1e]
            cropped_data[start:end, ds[1] / 2 - c0size0:ds[1] / 2 + c0size1,
            ds[2] / 2 - c1size0:ds[2] / 2 + c1size1] = crop * cropped_correction_factor
            gc.collect()

        cropped_valid_mask = np.zeros((M, M), dtype=np.float32)
        crop = self.valid_mask[c0s:c0e, c1s:c1e]
        cs = crop.shape
        print('crop.shape', cs)
        cropped_valid_mask[ds[1] / 2 - c0size0:ds[1] / 2 + c0size1, ds[2] / 2 - c1size0:ds[2] / 2 + c1size1] = crop

        self.correction_factor = None
        self.valid_mask = cropped_valid_mask
        self.data = cropped_data

    def get_valid_mask(self):
        return self.hot_pixel_mask  # * gap_mask

    def bin_data(self):
        if self.binning_factor > 1:
            print('Binning data by %d ...' % self.binning_factor)
            print('cropped_data.shape', self.data.shape)
            binned_data = u.rebin(self.data, (1, self.binning_factor, self.binning_factor))
            print('binned_data.shape ', binned_data.shape)
            #        io.plot(valid_mask,'valid_mask')
            valid_mask_rebin = u.rebin(self.valid_mask.astype(np.float32), (self.binning_factor, self.binning_factor),
                                       mode='mean')
            # io.plot(valid_mask_rebin, 'valid_mask_rebin')
            self.valid_mask = (valid_mask_rebin >= self.min_fraction_valid).astype(np.float32)
            # io.plot(valid_mask, 'valid_mask')
            # io.plot(binned_data[0], 'binned_data data')
            self.data = binned_data
            valid_mask_rebin = np.broadcast_to(self.valid_mask, self.data.shape)
            self.data[valid_mask_rebin > 0] /= valid_mask_rebin[valid_mask_rebin > 0]
            # io.plot(data[0], 'scaled binned_data data')
            # io.plot(data[0] * valid_mask, 'valid scaled data')

    def generate_valid_mask_nonzero_intensity(self):
        data_nonzero_mask = self.data > 0
        self.bvm = np.broadcast_to(self.valid_mask[np.newaxis, ...], self.data.shape)
        valid_mask_nonzero_intensity = np.logical_and(self.bvm, data_nonzero_mask)
        return valid_mask_nonzero_intensity

    def generate_valid_mask_nonzero_beam_valid(self):
        beam_mask_aperture = u.sector_mask((self.M / self.binning_factor, self.M / self.binning_factor),
                                           (self.M2 / self.binning_factor, self.M2 / self.binning_factor),
                                           self.radius_aperture / self.binning_factor, (0, 360))
        valid_mask_nonzero_beam_valid = self.valid_mask_nonzero_intensity.copy()
        self.bmb = np.broadcast_to(beam_mask_aperture[np.newaxis, ...], valid_mask_nonzero_beam_valid.shape)
        valid_mask_nonzero_beam_valid[self.bmb] = 1
        return valid_mask_nonzero_beam_valid

    def maybe_interpolate_dead_pixels(self):
        if self.interpolate_dead_pixels:
            print('Interpolating dead pixels...')
            kernel = Gaussian2DKernel(1)

            not_valid_and_inside_brightfield = np.logical_and(np.logical_not(self.bvm), self.bmb)
            self.data[not_valid_and_inside_brightfield] = np.NaN
            # p = Pool(multiprocessing.cpu_count())
            # p.map(replace_nans, data)
            for i, da in enumerate(self.data):
                self.data[i] = interpolate_replace_nans(da.copy(), kernel)

            self.data = np.nan_to_num(self.data, copy=False)

    def determine_positions(self):
        print('Creating position array ...')
        pixel_step_x = self.stepsize / self.dx
        pixel_step_y = self.stepsize / self.dx
        s = self.data.shape
        if self.binary_mask_file is None:
            pos = u.advanced_raster_scan(ny=self.stepy, nx=self.stepx, fast_axis=self.fast_axis, mirror=[1, 1], theta=0,
                                         dy=pixel_step_y, dx=pixel_step_x)
        else:
            X, Y = np.mgrid[0:self.stepx, 0:self.stepy]
            X = X.astype(np.float32)
            Y = Y.astype(np.float32)
            X *= pixel_step_x
            Y *= pixel_step_y
            x_pos = X[self.bin_mask_positions]
            y_pos = Y[self.bin_mask_positions]
            pos = np.zeros((s[0], 2))
            pos[:, 0] = y_pos
            pos[:, 1] = x_pos

        for ind in self.exclude_indices:
            pos = np.delete(pos, ind, 0)

        io.scatter_positions2(pos)

        return pos

    def prepare_initial_probe(self):
        print('Preparing initial probe ...')
        E = self.E_eV
        N = self.M / self.binning_factor
        defocus_nm = -self.df * 1e9
        det_pix = self.dpix
        alpha_rad = self.alpha_diff
        dx_angstrom = self.dx * 1e10
        print('defocus_nm :', defocus_nm)
        r, i, fr, fi = probe.focused_probe(E, N, d=dx_angstrom, alpha_rad=alpha_rad, defocus_nm=defocus_nm,
                                           det_pix=det_pix,
                                           C3_um=0, C5_mm=0, tx=0, ty=0, Nedge=2, plot=False)
        pr = (r + 1j * i).astype(np.complex128)
        fpr = ifft2(fftshift(pr), norm='ortho')

        io.plotAbsAngle(pr, 'probe real space')
        io.plotAbsAngle(fpr, 'probe aperture space')

        return pr, fpr

    def prepare_hdf5_dict(self):
        ret = u.Param()

        ret.mask = self.valid_mask.astype(np.float32)
        ret.mask_nonzero = self.valid_mask_nonzero_intensity.astype(np.float32)
        ret.mask_nonzero_beam_valid = self.valid_mask_nonzero_beam_valid
        ret.data = self.data
        #    data = None
        ret.alpha = self.alpha
        ret.alpha_diff = self.alpha_diff
        ret.z = self.z
        ret.E = self.E_eV
        ret.dpix = self.dpix
        ret.dx = self.dx
        ret.theta = 0
        ret.I_beam = self.beam_intensity
        ret.r_aperture = self.radius_aperture / self.binning_factor
        ret.center = self.c1[1:]
        ret.stepsize = self.stepsize
        ret.stepx = self.stepx
        ret.stepy = self.stepy
        ret.pos = self.pos
        ret.probe = self.pr.astype(np.complex64)
        ret.probe_fourier = self.fpr.astype(np.complex64)
        return ret

    def prepare_mat_dict(self):
        ret = u.Param()

        ret.mask = self.valid_mask.astype(np.float32)
        ret.mask_nonzero = self.valid_mask_nonzero_intensity.astype(np.float32)
        ret.mask_nonzero_beam_valid = self.valid_mask_nonzero_beam_valid
        ret.data = self.data
        #    data = None
        ret.alpha = self.alpha
        ret.alpha_diff = self.alpha_diff
        ret.z = self.z
        ret.E = self.E_eV
        ret.dpix = self.dpix
        ret.dx = self.dx
        ret.theta = 0
        ret.I_beam = self.beam_intensity
        ret.r_aperture = self.radius_aperture / self.binning_factor
        ret.center = self.c1[1:]
        ret.stepsize = self.stepsize
        ret.stepx = self.stepx
        ret.stepy = self.stepy
        ret.pos = self.pos
        ret.probe_real = self.pr.real.astype(np.float32)
        ret.probe_imag = self.pr.imag.astype(np.float32)
        ret.probe_fourier_real = self.fpr.real.astype(np.float32)
        ret.probe_fourier_imag = self.fpr.imag.astype(np.float32)
        return ret

    def prepare_dataset(self):
        r"""Prepares data for reconstruction

        Args:
            path (string): path to the data file
            name (string): name of the h5 file and the json file
            mask_file (string): path and name of the hot pixel mask
            step_size (float): real space step size
            q_max_rel (float): maximum scattering angle relative to diffraction limit angle (default: 1.1)
        """

        h5_filename = self.path + self.name + '.nxs'
        json_filename = self.path + self.name + '.json'

        json_dict = json.load(open(json_filename))
        #    print(json_dict)

        post_magnification = 1.56
        self.z = json_dict['Projection']['CameraLength'] * post_magnification
        self.E_eV = json_dict['Gun']['HTValue']
        self.lam = u.lam(self.E_eV)
        self.dpix = 55e-6 * self.binning_factor

        self.stepy = json_dict['Scanning']['Parameters']['Frame (Y)']['Steps']
        self.stepx = json_dict['Scanning']['Parameters']['Line (X)']['Steps']
        self.stepsize = json_dict['Illumination']['STEMSize'] / self.stepx * 1e-9
        self.df = json_dict['Projection']['Defocus']

        # f = h5py.File(h5_filename, "r")
        f = io.h5read(h5_filename)
        self.data = f['entry']['instrument']['detector']['data']
        s = np.array(self.data.shape)

        self.bin_mask_positions = self.load_binary_position_mask()
        self.hot_pixel_mask = self.load_hot_pixel_mask(s)
        self.correction_factor = self.load_correction_factor(s)
        self.valid_mask = self.get_valid_mask()

        x = np.linspace(0, self.stepx, endpoint=False, num=self.stepx).astype(np.int)
        y = np.linspace(0, self.stepy, endpoint=False, num=self.stepy).astype(np.int)
        yy, xx = np.meshgrid(y, x)
        xpos = xx[self.bin_mask_positions]
        ypos = yy[self.bin_mask_positions]

        for ind in self.exclude_indices:
            data = np.delete(self.data, ind, 0)
            # self.bin_mask_positions[xpos[ind], ypos[ind]] = 0
            self.bin_mask_positions[ypos[ind], xpos[ind]] = 0

        for ind in self.exclude_indices:
            xpos = np.delete(xpos, ind, 0)
            ypos = np.delete(ypos, ind, 0)

        s = np.array(self.data.shape)

        print('z            = {}m'.format(si_format(self.z)))
        print('E            = {}eV'.format(si_format(self.E_eV)))
        print('lam          = {}m'.format(si_format(self.lam, precision=2)))
        print('dpix         = {}m'.format(si_format(self.dpix, precision=2)))
        print('dataset size = ', s)

        self.prepare_stem_image()
        self.determine_center_rotation_alpha()

        self.M_diff = int(self.radius) * 2
        self.M = int(self.q_max_rel * self.radius) * 2
        self.M = self.M if self.M % (self.binning_factor * 2) == 0 else self.M + (self.M % (self.binning_factor * 2))
        self.M2 = self.M / 2

        print('M = {}'.format(self.M))

        dx = u.real_space_resolution(self.E_eV, self.z, self.dpix, self.M / self.binning_factor)
        dx_diff = u.real_space_resolution(self.E_eV, self.z, self.dpix, self.M_diff / self.binning_factor)
        alpha = np.arcsin(self.lam / (2 * dx))
        alpha_diff = np.arcsin(self.lam / (2 * dx_diff))
        self.dx = dx
        self.dx_diff = dx_diff
        self.alpha = alpha
        self.alpha_diff = alpha_diff

        self.pr, self.fpr = self.prepare_initial_probe()

        print('dx   = {}m'.format(si_format(dx, precision=2)))
        print('dx BF limit   = {}m'.format(si_format(dx_diff, precision=2)))
        print('alpha= {}rad'.format(si_format(alpha, precision=2)))
        print('alpha BF limit= {}rad'.format(si_format(alpha_diff, precision=2)))

        self.c = [self.c1[1], self.c1[2]]

        self.crop_data()
        self.bin_data()

        self.valid_mask_nonzero_intensity = self.generate_valid_mask_nonzero_intensity()
        self.valid_mask_nonzero_beam_valid = self.generate_valid_mask_nonzero_beam_valid()

        # io.plot(valid_mask * data[0], 'test cropped valid mask 1')
        # io.plot(valid_mask_nonzero_intensity[0] * data[0], 'test cropped valid mask 2')

        self.maybe_interpolate_dead_pixels()

        intensities = np.sum(self.data * self.valid_mask, (1, 2))
        max_intensity = intensities.max()
        max_intensity_ind = intensities.argmax()

        # io.plot(data[max_intensity_ind] * valid_mask_nonzero_intensity[max_intensity_ind], 'dp with maximum intensity')
        print('maximum intensity: %g at index %d' % (max_intensity, max_intensity_ind))

        beam_mask = u.sector_mask((self.M / self.binning_factor, self.M / self.binning_factor),
                                  (self.M2 / self.binning_factor, self.M2 / self.binning_factor),
                                  self.radius / self.binning_factor, (0, 360))
        # io.plot(beam_mask, 'beam_mask')
        mean_beam_pixel_intensity = np.mean(self.data[max_intensity_ind] * beam_mask * self.valid_mask)
        print('mean_beam_pixel_intensity: %g' % mean_beam_pixel_intensity)

        dead_pixels_in_BF_mask = np.logical_and(np.logical_not(self.valid_mask), beam_mask)
        # io.plot(dead_pixels_in_BF_mask, 'dead_pixels_in_BF_mask')
        dead_pixels_in_BF = np.sum(dead_pixels_in_BF_mask)
        print('dead_pixels_in_BF: %d' % dead_pixels_in_BF)

        self.beam_intensity = np.sum(
            self.data[max_intensity_ind] * self.valid_mask) + dead_pixels_in_BF * mean_beam_pixel_intensity
        print('beam_intensity: %g' % self.beam_intensity)

        self.pos = self.determine_positions()

        self.fpr /= m.sqrt(norm(self.fpr) ** 2)
        self.fpr *= m.sqrt(self.beam_intensity)

        self.pr /= m.sqrt(norm(self.pr) ** 2)
        self.pr *= m.sqrt(self.beam_intensity)

        if self.save_hdf5:
            print('Saving to hdf5 file format ...')
            ret = self.prepare_hdf5_dict()
            fname = '%s%s_bin%d_%s.h5' % (self.path, self.name, self.binning_factor, self.save_suffix)
            io.h5write(fname, ret)

        if self.save_matlab:
            print('Saving to matlab file format ...')
            from scipy.io import savemat
            ret = self.prepare_mat_dict()
            fname = '%s%s_bin%d_%s' % (self.path, self.name, self.binning_factor, self.save_suffix)
            savemat(fname, ret, do_compression=True)
        return ret


p = u.Param()

p.path = './'

p.name = 'Reg3_00004'
p.mask_file = './Ref12bit_pxmask.tif'
p.reference_file = './Ref12bit_reference.mrc'
p.binning_factor = 2
p.q_max_rel = 1.2
p.exclude_indices = []
p.min_fraction_valid = 0.49
p.interpolate_dead_pixels = True
p.binary_mask_file = None#'bin_array_02.tif'
p.save_hdf5 = True
p.save_matlab = True
p.save_suffix = 'processed'
p.fast_axis = 1

prep = DataPreparation(**p)
d = prep.prepare_dataset()
