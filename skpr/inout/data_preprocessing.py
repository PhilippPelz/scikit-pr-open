#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 16:45:00 2018

@author: pelzphil
"""

from __future__ import print_function

import json
import math as m
import time
import datetime
import mrcfile
import numpy as np
import pytiff
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
from numpy.fft import ifft2, fftshift
from numpy.linalg import norm
from si_prefix import si_format
from scipy import ndimage as ni

import h5rw as rw
import plot as io
import skpr.util as u
from skpr.simulation import probe
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
import dask.array as da
from dask import compute, delayed
import dask.threaded
import multiprocessing
import h5py


class DataPreparation:
    def __init__(self, path='./', save_path='./', name='R4_00009', mask_file='./Ref12bit_pxmask.tif', \
                 reference_file='./Ref12bit_reference.mrc', q_max_rel=1.1, data_size=None, exclude_indices=[], \
                 binning_factor=1, min_fraction_valid=0.5, interpolate_dead_pixels=True, binary_mask_file=None,
                 save_suffix='processed', fast_axis=1, save_hdf5=True, save_matlab=False,
                 experiment_geometry_entry='auto', experiment_geometry=None, theta=0, select_area=False,
                 selected_area_start=[0, 0], selected_area_size=64, file_extension='.nxs',
                 data_entry='/entry/instrument/detector/data', mirror=[1, 1], defocus_auto=True, \
                 mask_from_varmean=False, varmean_tolerance=0.1, dp_centering_method='linear', metadata_file=None,
                 do_plot=True, blur_stem=0, manual_data_selection=False, clip_interactive_data=True, cpu_count=None,
                 gap_mask_file=None, vacuum_measurements=None):
        self.manual_data_selection = manual_data_selection
        self.blur_stem = blur_stem
        self.path = path
        self.save_path = save_path
        self.name = name
        self.mask_file = mask_file
        self.gap_mask_file = gap_mask_file
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
        self.experiment_geometry_entry = experiment_geometry_entry
        self.experiment_geometry = experiment_geometry
        self.theta = theta
        self.select_area = select_area
        self.selected_area_start = selected_area_start
        self.selected_area_size = selected_area_size
        self.file_extension = file_extension
        self.data_entry = data_entry
        self.mirror = mirror
        self.defocus_auto = defocus_auto
        self.mask_from_varmean = mask_from_varmean
        self.varmean_tol = varmean_tolerance
        self.data_size = data_size
        self.metadata_file = metadata_file
        self.do_plot = do_plot
        self.clip_interactive_data = clip_interactive_data
        self.cpu_count = multiprocessing.cpu_count() if cpu_count is None else cpu_count
        self.vacuum_measurements = vacuum_measurements

        self.radius = np.array([250])
        self.radius_aperture = np.array([250])
        self.radius_aperture_inner = np.array([250])
        self.c1 = np.array([0, 0, 0])
        self.c2 = np.array([0, 0, 0])
        self.c3 = np.array([0, 0, 0])
        self.pos1 = np.array([0, 0, 0])
        self.pos2 = np.array([0, 0, 0])
        self.dp_centering_method = dp_centering_method

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
                bin_mask_positions = im.astype(np.bool)
            if self.do_plot:
                io.plot(bin_mask_positions.astype(np.int), 'binary position mask')
        else:
            bin_mask_positions = np.ones((self.stepy, self.stepx)).astype(np.bool)
        # bin_mask_positions = np.transpose(bin_mask_positions)
        print('Binary position mask size   :', bin_mask_positions.shape)
        print('Binary position mask entries: %d' % bin_mask_positions.sum())
        return bin_mask_positions

    def load_hot_pixel_mask(self):
        s = self.data.shape
        if self.mask_file is not None:
            print('Loading hot pixel mask ...')
            with pytiff.Tiff(self.mask_file) as handle:
                im = np.abs(np.array(handle))
                # print(im.dtype)
                # if self.do_plot:
                #     io.plot(im, 'hot pixel mask file')
                mask = (1 - im / im.max()).astype(np.int8)
            if self.do_plot:
                io.plot(mask, 'hot pixel mask')
            mask = mask[self.com[0] - self.rr:self.com[0] + self.rr, self.com[1] - self.rr:self.com[1] + self.rr].copy()
        else:
            mask = np.ones((s[1], s[2]))
            print(mask.shape, s)
        return np.broadcast_to(mask[np.newaxis, ...], s)

    def load_gap_mask(self):
        s = self.data.shape
        if self.gap_mask_file is not None:
            print('Loading gap pixel mask ...')
            with pytiff.Tiff(self.gap_mask_file) as handle:
                im = np.abs(np.array(handle))
                # print(im.dtype)
                # if self.do_plot:
                #     io.plot(im, 'hot pixel mask file')
                mask = (1 - im / im.max()).astype(np.int8)
            if self.do_plot:
                io.plot(mask, 'gap pixel mask')
            mask = mask[self.com[0] - self.rr:self.com[0] + self.rr, self.com[1] - self.rr:self.com[1] + self.rr].copy()
        else:
            mask = np.ones((s[1], s[2]))
        return np.broadcast_to(mask[np.newaxis, ...], s)

    def load_correction_factor(self, s):
        print('Loading correction factors ...')
        if self.reference_file is not None:
            filename, file_extension = os.path.splitext(self.reference_file)
            if file_extension == '.mrc':
                with mrcfile.open(self.reference_file) as mrc:
                    correction_factor = 1 / mrc.data
            else:
                with pytiff.Tiff(self.reference_file) as handle:
                    im = np.array(handle)
                    correction_factor = 1 / im
            correction_factor = correction_factor[self.com[0] - self.rr:self.com[0] + self.rr,
                                self.com[1] - self.rr:self.com[1] + self.rr].copy()
        else:
            correction_factor = np.ones((s[1], s[2]))
        return correction_factor

    def prepare_stem_image(self):
        print('Preparing STEM image ...')
        s = self.data.shape
        bf_mask = u.sector_mask((self.M / self.binning_factor, self.M / self.binning_factor),
                                (self.M2 / self.binning_factor, self.M2 / self.binning_factor),
                                self.radius_aperture_inner / self.binning_factor, (0, 360))
        df_mask = np.logical_not(bf_mask)
        data_split = np.array_split(self.data, self.cpu_count, 0)
        bf_stem_list = [delayed(lambda x, y: np.sum(x * y, (1, 2)))(x, bf_mask) for x in data_split]
        l = compute(*bf_stem_list, scheduler='threads')
        bf_stem_intensities = np.concatenate(l, 0)
        print('bf_stem_intensities.shape', bf_stem_intensities.shape)
        df_stem_list = [delayed(lambda x, y: np.sum(x * y, (1, 2)))(x, df_mask) for x in data_split]
        l = compute(*df_stem_list, scheduler='threads')
        df_stem_intensities = np.concatenate(l, 0)
        print('df_stem_intensities.shape', bf_stem_intensities.shape)
        # intensities = np.sum(self.data * beam_mask_aperture, (1, 2))

        if self.do_plot:
            num = np.linspace(0, len(bf_stem_intensities) - 1, len(bf_stem_intensities), endpoint=True)
            f, a = plt.subplots(figsize=(8, 20))
            a.scatter(num, bf_stem_intensities, s=1, marker='x')
            ss = '%s%s_%s_intensities' % (self.save_path, self.name, self.save_suffix)
            f.savefig(ss + '.png', dpi=600)
            plt.show()

        if self.binary_mask_file is None:
            BF_STEM = np.ones((self.stepy, self.stepx)).astype(np.float32) * np.mean(bf_stem_intensities)
            BF_STEM.flat[:len(bf_stem_intensities)] = bf_stem_intensities.flat

            DF_STEM = np.ones((self.stepy, self.stepx)).astype(np.float32) * np.mean(df_stem_intensities)
            DF_STEM.flat[:len(df_stem_intensities)] = df_stem_intensities.flat
        else:
            from scipy.ndimage.filters import gaussian_filter

            ind = np.linspace(0, s[0], endpoint=False, num=s[0]).astype(np.int)

            indices = np.zeros((self.stepy, self.stepx)).astype(np.int)
            indices[self.bin_mask_positions] = ind
            # io.plot(indices,'indices')
            BF_STEM = np.zeros((self.stepy, self.stepx)).astype(np.float32)
            BF_STEM[:] = np.mean(bf_stem_intensities)
            BF_STEM[self.bin_mask_positions] = bf_stem_intensities

            if self.blur_stem > 0:
                BF_STEM = gaussian_filter(BF_STEM, sigma=self.blur_stem)

            DF_STEM = np.zeros((self.stepy, self.stepx)).astype(np.float32)
            DF_STEM[:] = np.mean(df_stem_intensities)
            DF_STEM[self.bin_mask_positions] = df_stem_intensities
            if self.blur_stem > 0:
                DF_STEM = gaussian_filter(DF_STEM, sigma=self.blur_stem)

        unit = BF_STEM.shape[0] // 10 * self.stepsize
        sc = (BF_STEM.shape[0] // 10, '{}m'.format(si_format(unit, precision=2)))

        io.plot(BF_STEM, 'BF-STEM', savePath='%s%s_%s_BF-STEM' % (self.save_path, self.name, self.save_suffix),
                show=self.do_plot, scale=sc)
        io.plot(DF_STEM, 'DF-STEM', savePath='%s%s_%s_DF-STEM' % (self.save_path, self.name, self.save_suffix),
                show=self.do_plot, scale=sc)

    def determine_center_rotation_alpha(self):
        if self.metadata_file is None:
            print('Determine center, alpha, and rotation angle ...')

            r = self.data.shape[1] / 2 * 0.8

            action_sequence = [
                ('Please match the radius of the diffraction disc CONTROL', 'control', 'r', self.radius),
                ('Please match the outer rim radius of the aperture CONTROL', 'control', 'r', self.radius_aperture),
                ('Please match the inner rim radius of the aperture CONTROL', 'control', 'r',
                 self.radius_aperture_inner),
                ('Now determine the center of the disc with your cursor ENTER', 'enter', 'pos', self.c1),
                ('Closing', 'close', 'pos', self.pos2)
            ]
            show_it = self.data[[0]]
            show_it *= self.hot_pixel_mask[0]
            show_it = np.clip(show_it, 0, 5)
            cursor = u.InteractiveDataPrep(show_it, r, action_sequence)

            action_sequence = [
                (
                    'Now determine the center again. ENTER', 'enter', 'pos', self.c2),
                ('Closing', 'close', 'pos', self.pos2)
            ]
            show_it = self.data[[self.stepx - 1]]
            show_it *= self.hot_pixel_mask[0]
            show_it = np.clip(show_it, 0, 5)
            cursor = u.InteractiveDataPrep(show_it, self.radius, action_sequence)

            action_sequence = [
                (
                    'Now determine the center again. ENTER', 'enter', 'pos', self.c3),
                ('Closing', 'close', 'pos', self.pos2)
            ]
            show_it = self.data[[(self.stepy - 1) * self.stepx]]
            show_it *= self.hot_pixel_mask[0]
            show_it = np.clip(show_it, 0, 5)
            cursor = u.InteractiveDataPrep(show_it, self.radius, action_sequence)
        else:
            print('Loading centers and radius from metadata file %s' % self.metadata_file)
            bin = rw.h5read(self.metadata_file, 'binning_factor').values()[0]
            self.radius = rw.h5read(self.metadata_file, 'r').values()[0] * bin
            self.radius_aperture = rw.h5read(self.metadata_file, 'r_aperture').values()[0] * bin
            self.radius_aperture_inner = rw.h5read(self.metadata_file, 'r_aperture_inner').values()[0] * bin

        print('r          = {}'.format(self.radius))
        print('r_aperture = {}'.format(self.radius_aperture))
        print('r_inner    = {}'.format(self.radius_aperture_inner))

    def crop_data(self):
        M = self.M
        M2 = self.M2
        s = self.data.shape
        ds = (s[0], M, M)

        def crop_data1(d, c, corr_fac, ds):
            cropped_data = np.zeros((d.shape[0], M, M), dtype=np.float32)
            for i in range(d.shape[0]):
                c0s = c[i, 0] - M2 if (c[i, 0] - M2) > 0 else 0
                c1s = c[i, 1] - M2 if (c[i, 1] - M2) > 0 else 0
                c0e = c[i, 0] + M2 if (c[i, 0] + M2) < s[1] else s[1]
                c1e = c[i, 1] + M2 if (c[i, 1] + M2) < s[2] else s[2]

                c0size0 = M2 if (c[i, 0] - M2) > 0 else c[i, 0]
                c1size0 = M2 if (c[i, 1] - M2) > 0 else c[i, 1]
                c0size1 = M2 if (c[i, 0] + M2) < s[1] else s[1] - c[i, 0]
                c1size1 = M2 if (c[i, 1] + M2) < s[2] else s[2] - c[i, 1]

                cropped_correction_factor = corr_fac[i, c0s:c0e, c1s:c1e]
                crop = d[i, c0s:c0e, c1s:c1e]

                cropped_data[i, ds[1] / 2 - c0size0:ds[1] / 2 + c0size1,
                ds[2] / 2 - c1size0:ds[2] / 2 + c1size1] = crop.astype(np.float32) * cropped_correction_factor

            return cropped_data

        print('Cropping data ...')
        data_split = np.array_split(self.data, self.cpu_count, 0)
        c_split = np.array_split(self.c.astype(np.int), self.cpu_count, 0)
        bc_corrfac = np.broadcast_to(self.correction_factor[np.newaxis, ...], self.data.shape)
        bc_corrfac_split = np.array_split(bc_corrfac, self.cpu_count, 0)
        crop_data_compute_list = [delayed(crop_data1)(d, c, corr_fac, ds) for d, c, corr_fac in
                                  zip(data_split, c_split, bc_corrfac_split)]
        cropped_data = compute(*crop_data_compute_list, scheduler='threads')

        self.correction_factor = None
        self.data = cropped_data

    def get_cropped_valid_mask(self):
        if self.mask_from_varmean:
            print('Taking valid mask from var and mean of data')
            v = np.var(self.data, 0)
            m = np.mean(self.data, 0)
            vm = v / m
            io.plot(vm, 'var/mean')
            mask = np.logical_and(vm > 1 - self.varmean_tol, vm < 1 + self.varmean_tol)
            io.plot(mask.astype(np.float32), '%g < var/mean < %g' % (1 - self.varmean_tol, 1 + self.varmean_tol))
            self.valid_mask = np.broadcast_to(mask, self.data.shape)
            self.valid_mask = np.array_split(self.valid_mask, self.cpu_count, 0)
        else:
            print('Taking valid mask from hot pixel mask')
            M = self.M
            M2 = self.M2
            s = self.hot_pixel_mask.shape
            ds = (s[0], M, M)

            def crop_mask(vm, c, ds):
                cvm = np.zeros((vm.shape[0], M, M), dtype=np.int8)
                for i in range(vm.shape[0]):
                    c0s = c[i, 0] - M2 if (c[i, 0] - M2) > 0 else 0
                    c1s = c[i, 1] - M2 if (c[i, 1] - M2) > 0 else 0
                    c0e = c[i, 0] + M2 if (c[i, 0] + M2) < s[1] else s[1]
                    c1e = c[i, 1] + M2 if (c[i, 1] + M2) < s[2] else s[2]

                    c0size0 = M2 if (c[i, 0] - M2) > 0 else c[i, 0]
                    c1size0 = M2 if (c[i, 1] - M2) > 0 else c[i, 1]
                    c0size1 = M2 if (c[i, 0] + M2) < s[1] else s[1] - c[i, 0]
                    c1size1 = M2 if (c[i, 1] + M2) < s[2] else s[2] - c[i, 1]

                    crop = vm[i, c0s:c0e, c1s:c1e]
                    cvm[i, ds[1] / 2 - c0size0:ds[1] / 2 + c0size1,
                    ds[2] / 2 - c1size0:ds[2] / 2 + c1size1] = crop

                return cvm

            print('Cropping mask ...')
            hot_pixel_mask_split = np.array_split(self.hot_pixel_mask, self.cpu_count, 0)
            gap_mask_split = np.array_split(self.gap_mask, self.cpu_count, 0)
            c_split = np.array_split(self.c.astype(np.int), self.cpu_count, 0)
            crop_mask_compute_list = [delayed(crop_mask)(d, c, ds) for d, c in
                                      zip(hot_pixel_mask_split, c_split)]
            cropped_valid_mask = compute(*crop_mask_compute_list, scheduler='threads')

            crop_gap_mask_compute_list = [delayed(crop_mask)(d, c, ds) for d, c in
                                          zip(gap_mask_split, c_split)]
            cropped_gap_mask = compute(*crop_gap_mask_compute_list, scheduler='threads')
            self.valid_mask = cropped_valid_mask
            self.gap_mask = cropped_gap_mask

    def bin_data(self):
        if self.binning_factor > 0:
            print('Binning data by %d ...' % self.binning_factor)

            def mul_rebin(x, y):
                d = x * y
                ret = u.rebin(d, (1, self.binning_factor, self.binning_factor), mode='sum')
                return ret

            data_split = self.data
            vm_split = self.valid_mask
            binned_data_compute_list = [delayed(mul_rebin)(*z) for
                                        z in zip(data_split, vm_split)]
            res = compute(*binned_data_compute_list, scheduler='threads')
            self.data = res
            if self.do_plot:
                io.plot(self.data[0][0], 'scaled binned data')

    def bin_mask(self):
        if self.binning_factor > 0:
            print('Binning mask by %d ...' % self.binning_factor)

            def q(x):
                ret = u.rebin(x, (1, self.binning_factor, self.binning_factor), mode='sum') / (
                    self.binning_factor * self.binning_factor)
                return ret

            vm_split = self.valid_mask
            valid_mask_rebin_compute_list = [delayed(q)(x.astype(np.float32)) for x in vm_split]
            self.valid_mask = compute(*valid_mask_rebin_compute_list, scheduler='threads')

            gm_split = self.gap_mask
            gap_mask_rebin_compute_list = [delayed(q)(x.astype(np.float32)) for x in gm_split]
            self.gap_mask = compute(*gap_mask_rebin_compute_list, scheduler='threads')
            if self.do_plot:
                io.plot(self.valid_mask[0][0], 'valid mask 0 after binning', savePath=self.save_path + 'binned_mask')
                io.plot(self.gap_mask[0][0], 'gap mask 0 after binning', savePath=self.save_path + 'binned_mask')

    def define_rebinned_mask(self):
        if self.binning_factor > 0:
            def q1(x, fraction):
                ret = (x >= fraction).astype(np.float32)
                return ret

            valid_mask_rebin_compute_list = [delayed(q1)(x, self.min_fraction_valid) for x in self.valid_mask]
            self.valid_mask = compute(*valid_mask_rebin_compute_list, scheduler='threads')

            gap_mask_rebin_compute_list = [delayed(q1)(x, self.min_fraction_valid) for x in self.gap_mask]
            self.gap_mask = compute(*gap_mask_rebin_compute_list, scheduler='threads')

            # valid_mask_compute_list = [delayed(lambda x, y: x*y)(vm,gm) for vm,gm in zip(self.valid_mask,self.gap_mask)]
            # self.valid_mask = np.vstack(compute(*valid_mask_compute_list, scheduler='threads'))
            self.valid_mask = np.vstack(self.valid_mask)

            if self.do_plot:
                io.plot(self.valid_mask[0], 'scaled binned mask')
            print('  valid_mask.shape ', self.valid_mask.shape)

    def correct_mask_scaling(self):
        if self.binning_factor > 0:
            print('Binning data by %d ...' % self.binning_factor)

            def mul_rebin(x, m):
                x[m > 0] /= m[m > 0]
                return x

            binned_data_compute_list = [delayed(mul_rebin)(d.astype(np.float32), v.astype(np.float32)) for
                                        d, v in zip(self.data, self.valid_mask)]
            self.data = np.vstack(compute(*binned_data_compute_list, scheduler='threads'))
            print('cropped_data.shape ', self.data.shape)

    def generate_valid_mask_nonzero_intensity(self):
        data_nonzero_mask = self.data > 0
        self.bvm = self.valid_mask
        valid_mask_nonzero_intensity = np.logical_and(self.bvm, data_nonzero_mask)
        return valid_mask_nonzero_intensity

    def generate_valid_mask_beam_valid(self):
        beam_mask_aperture = u.sector_mask((self.M / self.binning_factor, self.M / self.binning_factor),
                                           (self.M2 / self.binning_factor, self.M2 / self.binning_factor),
                                           self.radius_aperture_inner / self.binning_factor, (0, 360))
        valid_mask_beam_valid = self.valid_mask.copy()
        self.bmb = np.broadcast_to(beam_mask_aperture[np.newaxis, ...], valid_mask_beam_valid.shape)
        valid_mask_beam_valid[self.bmb] = 1
        return valid_mask_beam_valid

    def maybe_interpolate_dead_pixels(self):
        not_valid_and_inside_brightfield = np.logical_and(np.logical_not(self.valid_mask), self.bmb)
        if self.interpolate_dead_pixels:
            print('Interpolating dead pixels...')
            kernel = Gaussian2DKernel(1)

            self.data[not_valid_and_inside_brightfield] = np.NaN
            # p = Pool(multiprocessing.cpu_count())
            # p.map(replace_nans, data)
            for i, data in enumerate(self.data):
                self.data[i] = interpolate_replace_nans(data.copy(), kernel)

            self.data = np.nan_to_num(self.data, copy=False)
        else:
            self.data[not_valid_and_inside_brightfield] = 0

    def determine_positions(self):
        print('Creating position array ...')
        print('stepsize ', self.stepsize)
        print('dx       ', self.dx)
        if self.experiment_geometry.pixel_stepx is None and self.experiment_geometry.pixel_stepy is None:
            pixel_step_x = self.stepsize / self.dx
            pixel_step_y = self.stepsize / self.dx
            self.pixel_step_x = pixel_step_x
            self.pixel_step_y = pixel_step_y
        else:
            self.pixel_step_x = self.experiment_geometry.pixel_stepx
            self.pixel_step_y = self.experiment_geometry.pixel_stepy

        print('pixel_step_x :', self.pixel_step_x)
        print('pixel_step_y :', self.pixel_step_y)
        s = self.data.shape
        if self.bin_mask_positions is None:
            print('Creating raster position array ...')
            pos = u.advanced_raster_scan(ny=self.stepy, nx=self.stepx, fast_axis=self.fast_axis, mirror=self.mirror,
                                         theta=self.theta,
                                         dy=self.pixel_step_y, dx=self.pixel_step_x)
            pos1 = u.advanced_raster_scan(ny=self.stepy, nx=self.stepx, fast_axis=self.fast_axis, mirror=self.mirror,
                                          theta=self.theta + 2,
                                          dy=self.pixel_step_y, dx=self.pixel_step_x)
            pos2 = u.advanced_raster_scan(ny=self.stepy, nx=self.stepx, fast_axis=self.fast_axis, mirror=self.mirror,
                                          theta=self.theta - 2,
                                          dy=self.pixel_step_y, dx=self.pixel_step_x)
            pos3 = u.advanced_raster_scan(ny=self.stepy, nx=self.stepx, fast_axis=self.fast_axis, mirror=self.mirror,
                                          theta=self.theta + 4,
                                          dy=self.pixel_step_y, dx=self.pixel_step_x)
            pos4 = u.advanced_raster_scan(ny=self.stepy, nx=self.stepx, fast_axis=self.fast_axis, mirror=self.mirror,
                                          theta=self.theta - 4,
                                          dy=self.pixel_step_y, dx=self.pixel_step_x)
        else:
            print('Creating position array from position mask...')
            X, Y = np.mgrid[0:self.stepx, 0:self.stepy]
            X = X.astype(np.float32)
            Y = Y.astype(np.float32)
            X *= self.pixel_step_x
            Y *= self.pixel_step_y
            x_pos = X[self.bin_mask_positions]
            y_pos = Y[self.bin_mask_positions]
            pos = np.zeros((s[0], 2))
            pos[:, 0] = y_pos
            pos[:, 1] = x_pos

            mins = np.array([pos[:, 0].min(), pos[:, 1].min()])
            maxs = np.array([pos[:, 0].max(), pos[:, 1].max()])

            center = mins + (maxs - mins) / 2.0
            pos -= center

            theta_rad = self.theta / 180.0 * np.pi
            R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                          [np.sin(theta_rad), np.cos(theta_rad)]])
            # rotate counterclockwise by theta
            pos = pos.dot(R)

            pos1 = pos
            pos2 = pos
            pos3 = pos
            pos4 = pos

        for ind in self.exclude_indices:
            pos = np.delete(pos, ind, 0)
            pos1 = np.delete(pos1, ind, 0)
            pos2 = np.delete(pos2, ind, 0)
            pos3 = np.delete(pos3, ind, 0)
            pos4 = np.delete(pos4, ind, 0)

        if self.do_plot:
            io.scatter_positions2(pos, show=self.do_plot, savePath='%s_pos' % self.name)

        return pos, pos1, pos2, pos3, pos4

    def prepare_initial_probe(self):
        E = self.E_eV
        N = self.M / self.binning_factor
        defocus_nm = self.df * 1e9
        det_pix = self.dpix
        alpha_rad = self.alpha_diff
        dx_angstrom = self.dx * 1e10

        print('Preparing initial probe ...')
        print('defocus_nm  :', defocus_nm)
        print('dx_angstrom :', dx_angstrom)
        print('alpha_rad   :', alpha_rad)
        print('defocus_nm  :', defocus_nm)
        print('det_pix     :', det_pix)

        probes = []
        fourier_probes = []

        r, i, fr, fi = probe.focused_probe(E, N, d=dx_angstrom, alpha_rad=alpha_rad, defocus_nm=defocus_nm,
                                           det_pix=det_pix, C3_um=2.2, C5_mm=0, tx=0, ty=0, Nedge=2, plot=False)
        pr = (r + 1j * i).astype(np.complex128)
        fpr = ifft2(fftshift(pr), norm='ortho')

        probes.append(pr)
        fourier_probes.append(fpr)

        for k in range(5):
            df = (defocus_nm + (k + 1) * 100)
            r, i, fr, fi = probe.focused_probe(E, N, d=dx_angstrom, alpha_rad=alpha_rad, defocus_nm=df,
                                               det_pix=det_pix, C3_um=2.2, C5_mm=0, tx=0, ty=0, Nedge=2, plot=False)
            pr = (r + 1j * i).astype(np.complex128)
            fpr = ifft2(fftshift(pr), norm='ortho')

            probes.append(pr)
            fourier_probes.append(fpr)

            df = (defocus_nm - (k + 1) * 100)
            r, i, fr, fi = probe.focused_probe(E, N, d=dx_angstrom, alpha_rad=alpha_rad, defocus_nm=df,
                                               det_pix=det_pix, C3_um=2.2, C5_mm=0, tx=0, ty=0, Nedge=2, plot=False)
            pr = (r + 1j * i).astype(np.complex128)
            fpr = ifft2(fftshift(pr), norm='ortho')

            probes.append(pr)
            fourier_probes.append(fpr)
        if self.do_plot:
            io.plotAbsAngle(probes[0], 'probe real space')
            io.plotAbsAngle(fourier_probes[0], 'probe aperture space')

        return np.array(probes), np.array(fourier_probes)

    def prepare_hdf5_dict(self):
        ret = u.Param()

        ret.mask = fftshift(self.valid_mask.astype(np.float32), (1, 2))
        ret.mask_beam_valid = fftshift(self.valid_mask_beam_valid.astype(np.float32), (1, 2))
        ret.data = fftshift(self.data, (1, 2))
        ret.alpha = self.alpha
        ret.alpha_diff = self.alpha_diff
        ret.z = self.z
        ret.E = self.E_eV
        ret.dpix = self.dpix
        ret.dx = self.dx
        ret.theta = 0
        ret.I_beam = self.beam_intensity
        ret.r_aperture = self.radius_aperture / self.binning_factor
        ret.r_aperture_inner = self.radius_aperture_inner / self.binning_factor
        ret.r = self.radius / self.binning_factor
        ret.centers = self.c
        ret.centers_residual = self.c_residual
        ret.stepsize = self.stepsize
        ret.stepx = self.stepx
        ret.stepy = self.stepy
        ret.pixel_step_y = self.pixel_step_y
        ret.pixel_step_x = self.pixel_step_x
        ret.pos = self.pos
        ret.pos1 = self.pos1
        ret.pos2 = self.pos2
        ret.pos3 = self.pos3
        ret.pos4 = self.pos4
        ret.probe = self.pr.astype(np.complex64)
        ret.probe_fourier = self.fpr.astype(np.complex64)
        ret.grid_positions = self.grid_positions
        ret.binning_factor = self.binning_factor
        if self.vacuum_measurements is not None:
            ret.vacuum_mean = self.vacuum_mean

        return ret

    def prepare_mat_dict(self):
        ret = u.Param()

        ret.mask = self.valid_mask.astype(np.float32)
        ret.mask_beam_valid = self.valid_mask_beam_valid.astype(np.float32)
        ret.data = self.data, (1, 2)
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
        ret.r_aperture_inner = self.radius_aperture_inner / self.binning_factor
        ret.r = self.radius / self.binning_factor
        ret.centers = self.c
        ret.centers_residual = self.c_residual
        ret.stepsize = self.stepsize
        ret.stepx = self.stepx
        ret.stepy = self.stepy
        ret.pixel_step_y = self.pixel_step_y
        ret.pixel_step_x = self.pixel_step_x
        ret.pos = self.pos
        ret.pos1 = self.pos1
        ret.pos2 = self.pos2
        ret.pos3 = self.pos3
        ret.pos4 = self.pos4
        ret.probe_real = self.pr.real.astype(np.float32)
        ret.probe_imag = self.pr.imag.astype(np.float32)
        ret.probe_fourier_real = self.fpr.real.astype(np.float32)
        ret.probe_fourier_imag = self.fpr.imag.astype(np.float32)
        ret.grid_positions = self.grid_positions
        ret.binning_factor = self.binning_factor
        if self.vacuum_measurements is not None:
            ret.vacuum_mean = self.vacuum_mean
        return ret

    def determine_dp_centers(self):
        if self.metadata_file is None:
            if self.dp_centering_method == 'linear':
                ctr_0 = np.array([self.c1[1], self.c1[2]], dtype=np.float32)
                ctr_end_col = np.array([self.c2[1], self.c2[2]], dtype=np.float32)
                ctr_end_row = np.array([self.c3[1], self.c3[2]], dtype=np.float32)
                dp_shift_per_column = (ctr_end_col - ctr_0) / (self.stepx - 1)
                dp_shift_per_row = (ctr_end_row - ctr_0) / (self.stepy - 1)
                print('ctr_0        ', ctr_0)
                print('ctr_end_col  ', ctr_end_col)
                print('ctr_end_row  ', ctr_end_row)
                print('dp_shift_per_column  ', dp_shift_per_column)
                print('dp_shift_per_row     ', dp_shift_per_row)

                ctr = np.zeros(((self.stepy * self.stepx), 2))
                ctr_res = np.zeros(((self.stepy * self.stepx), 2))

                for y in range(self.stepy):
                    for x in range(self.stepx):
                        ctr[y * self.stepx + x] = np.around(ctr_0 + x * dp_shift_per_column + y * dp_shift_per_row)
                        ctr_res[y * self.stepx + x] = (ctr_0 + x * dp_shift_per_column + y * dp_shift_per_row) - ctr[
                            y * self.stepx + x]

                # fx = interp1d([c1[0], c2[0]], [self.c1[1], self.c2[1]],
                #               fill_value='extrapolate')
                # fy = interp1d([c1[1], c2[1]], [self.c1[2], self.c2[2]],
                #               fill_value='extrapolate')
                #
                # # x = np.linspace(self.c1[1], self.c2[1], endpoint=True, num=self.stepx)
                # # y = np.linspace(self.c1[2], self.c2[2], endpoint=True, num=self.stepy)
                # x = fx(np.arange(self.stepx))
                # y = fy(np.arange(self.stepy))
                # xr = np.round(x)
                # yr = np.round(y)
                # cxx_int, cyy_int = np.meshgrid(xr, yr)
                # cxx, cyy = np.meshgrid(x, y)
                # cyy_residual = cyy - cyy_int
                # cxx_residual = cxx - cxx_int
                #
                # # io.plot(cxx_int,'cxx_int')
                # # io.plot(cyy_int, 'cyy_int')
                #
                # cy = cyy_int[self.bin_mask_positions]
                # cx = cxx_int[self.bin_mask_positions]
                #
                # cy_residual = cyy_residual[self.bin_mask_positions]
                # cx_residual = cxx_residual[self.bin_mask_positions]

                print('centers', ctr)

                self.c = ctr
                self.c_residual = ctr_res
                self.grid_positions = np.stack([self.xpos, self.ypos], -1)
            elif self.dp_centering_method == 'registration':
                pass
        else:
            self.c = rw.h5read(self.metadata_file, 'centers').values()[0]
            self.c_residual = rw.h5read(self.metadata_file, 'centers_residual').values()[0]
            self.grid_positions = rw.h5read(self.metadata_file, 'grid_positions').values()[0]

    def set_geometry_parameters(self):
        if self.experiment_geometry_entry == 'auto':
            print('Loading experiment geometry from metadata...')
            json_filename = self.path + self.name + '.json'
            json_dict = json.load(open(json_filename))
            post_magnification = 1.58
            z = json_dict['Projection']['CameraLength']
            if z == 1.0:
                z = 1.005
            elif z == 0.73:
                z = 0.707
            elif z == 0.52:
                z = 0.509
            self.z = z * post_magnification
            self.E_eV = json_dict['Gun']['HTValue']
            self.lam = u.lam(self.E_eV)
            self.dpix = 55e-6

            self.stepy = json_dict['Scanning']['Parameters']['Frame (Y)']['ROI len']
            self.stepx = json_dict['Scanning']['Parameters']['Line (X)']['ROI len']
            self.pointsx = json_dict['Scanning']['Parameters']['Line (X)']['Pts']
            self.pointsy = json_dict['Scanning']['Parameters']['Line (X)']['Pts']
            self.stepsize = json_dict['Illumination']['STEMSize'] / self.pointsy * 1e-9
            self.df = json_dict['Projection']['Defocus']
        else:
            print('Manual experiment geometry from metadata...')
            if self.experiment_geometry.alpha_diff is not None:
                self.alpha_diff = self.experiment_geometry.alpha_diff
            else:
                self.z = self.experiment_geometry.z
            self.E_eV = self.experiment_geometry.E_eV
            self.lam = u.lam(self.E_eV)
            self.dpix = self.experiment_geometry.dpix

            self.stepy = self.experiment_geometry.stepy
            self.stepx = self.experiment_geometry.stepx
            self.stepsize = self.experiment_geometry.stepsize
            self.df = self.experiment_geometry.df

        if not self.defocus_auto:
            self.df = self.experiment_geometry.df

        self.dpix *= self.binning_factor

    def print_timestamp(self):
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        print(st)

    def prepare_dataset(self):
        r"""Prepares data for reconstruction

        Args:
            path (string): path to the data file
            name (string): name of the h5 file and the json file
            mask_file (string): path and name of the hot pixel mask
            step_size (float): real space step size
            q_max_rel (float): maximum scattering angle relative to diffraction limit angle (default: 1.1)
        """

        h5_filename = self.path + self.name + self.file_extension

        self.print_timestamp()
        self.set_geometry_parameters()

        print('Loading data from file %s' % h5_filename)
        f = h5py.File(h5_filename, 'r')
        # self.data = rw.h5read(h5_filename, self.data_entry).values()[0]

        d0 = f[self.data_entry]
        d0s = np.array(d0.shape)
        d0s[0] = d0s[0] // 20
        print('d0s,', tuple(d0s))
        self.data = da.from_array(d0, chunks=tuple(d0s))
        s = np.array(self.data.shape)
        print('data type              ', self.data.dtype)
        self.print_timestamp()
        print('initial dataset size = ', s)
        print('Data loaded.')
        self.bin_mask_positions = self.load_binary_position_mask()

        if self.manual_data_selection:
            dsum = da.sum(self.data[100:200], 0).compute()

            smallest_side = np.min(s[1:] // 2)
            print(smallest_side)
            cm = ni.center_of_mass(dsum)
            self.com = com = np.array(cm).astype(np.int)
            self.rr = rr = np.min(np.array([com[0], com[1], s[1] - com[0], s[2] - com[1], smallest_side]))
            print('com,radius = ', com, rr)

            dcrop = self.data[:, com[0] - rr:com[0] + rr, com[1] - rr:com[1] + rr].compute()
            self.data = dcrop
            dcrop1 = da.from_array(dcrop, chunks=dcrop.shape)
            dcsum = da.sum(dcrop1, (1, 2)).compute()

            def get_com(imgs):
                ret = np.ones((imgs.shape[0]))
                for i, dd in enumerate(imgs):
                    comx1, comy1 = ni.center_of_mass(dd)
                    ret[i] = comy1
                return ret

            dcrop_split = np.array_split(dcrop, self.cpu_count, 0)
            com_compute_list = [delayed(get_com)(d) for d in dcrop_split]
            comy = np.hstack(compute(*com_compute_list, scheduler='threads'))
            print('comy.shape ', comy.shape)
            global f
            f, a = plt.subplots(figsize=(20, 8))
            a.scatter(np.arange(len(comy)), comy, s=1)

            idx = np.array([0, 0])

            def onclick(event):
                global f
                # print(event.button)
                if event.button == 3:
                    ix, iy = event.xdata, event.ydata

                    idx[0] = ix

                    f.canvas.mpl_disconnect(cid)
                    plt.close(f)

            cid = f.canvas.mpl_connect('button_press_event', onclick)

            plt.show()
            idx[1] = idx[0] + (self.stepx * self.stepy)
            print('start, end = ', idx)

            self.data = self.data[idx[0]:idx[1], ...]
            dcsum = dcsum[idx[0]:idx[1]]
            dcsum1 = dcsum.reshape((self.stepy, self.stepx))

            s = np.array(self.data.shape)
            print('loaded dataset size = ', s)
            x = u.MaskPrep(dcsum1, np.ones_like(dcsum1))
            self.bin_mask_positions = x.current_mask.astype(np.bool)
            if self.do_plot:
                io.plot(self.bin_mask_positions.astype(np.int), 'valid positions')
        else:
            self.data = self.data.compute()

        if self.select_area:
            size = self.selected_area_size
            print('Selecting an area of size %d x %d ...' % (size, size))
            st = self.selected_area_start
            print('reshape to ', (self.stepy, self.stepx, s[1], s[2]))
            # d1 = np.reshape(self.data, (self.stepy, self.stepx, s[1], s[2]))
            d1 = self.data.reshape((self.stepy, self.stepx, s[1], s[2]))

            self.data = d1[st[0]:st[0] + size, st[1]:st[1] + size, ...]
            self.bin_mask_positions = self.bin_mask_positions[st[0]:st[0] + size, st[1]:st[1] + size]

            ds = self.data.shape
            self.data = self.data.reshape((ds[0] * ds[1], ds[2], ds[3]))
            s = np.array(self.data.shape)
            print('New data shape after selecting area: ', s)
            self.stepx = self.stepy = size

        x = np.linspace(0, self.stepx, endpoint=False, num=self.stepx).astype(np.int)
        y = np.linspace(0, self.stepy, endpoint=False, num=self.stepy).astype(np.int)
        yy, xx = np.meshgrid(x, y)
        self.xpos = xx[self.bin_mask_positions]
        self.ypos = yy[self.bin_mask_positions]

        bmflat = self.bin_mask_positions.flatten()
        self.data = self.data[bmflat]
        self.bin_mask_positions_flat = bmflat

        s = np.array(self.data.shape)

        print('dataset size after excluding indices = ', s)
        self.hot_pixel_mask = self.load_hot_pixel_mask()
        self.gap_mask = self.load_gap_mask()

        self.determine_center_rotation_alpha()

        self.M_diff = int(self.radius * 2)

        if self.data_size is None:
            self.M = int(self.q_max_rel * self.radius * 2)
        else:
            self.M = self.data_size

        self.M = self.M if self.M % (self.binning_factor * 2) == 0 else self.M - (self.M % (self.binning_factor * 2))
        self.M2 = self.M / 2

        print('                           M = {}'.format(self.M))
        print('after considering binning: M = {}'.format(self.M))

        if self.experiment_geometry.alpha_diff is not None:
            self.z = (self.radius * self.dpix / self.binning_factor / np.tan(self.alpha_diff))[0]
            self.alpha = np.arctan(self.M / self.binning_factor / 2 * self.dpix / self.z)
        else:
            self.alpha_diff = np.arctan(self.M_diff / self.binning_factor / 2 * self.dpix / self.z)
            self.alpha = np.arctan(self.M / self.binning_factor / 2 * self.dpix / self.z)

        self.dx = u.real_space_resolution(self.E_eV, self.z, self.dpix, self.M / self.binning_factor)
        self.dx_diff = u.real_space_resolution(self.E_eV, self.z, self.dpix, self.M_diff / self.binning_factor)

        self.pr, self.fpr = self.prepare_initial_probe()

        print('z              = {}m'.format(si_format(self.z)))
        print('E              = {}eV'.format(si_format(self.E_eV)))
        print('lam            = {}m'.format(si_format(self.lam, precision=2)))
        print('det_pix        = {}m'.format(si_format(self.dpix, precision=2)))
        print('dx             = {}m'.format(si_format(self.dx, precision=2)))
        print('dx BF limit    = {}m'.format(si_format(self.dx_diff, precision=2)))
        print('alpha          = {}rad'.format(si_format(self.alpha, precision=2)))
        print('alpha BF limit = {}rad'.format(si_format(self.alpha_diff, precision=2)))

        self.determine_dp_centers()

        self.correction_factor = self.load_correction_factor(s)

        self.get_cropped_valid_mask()

        self.crop_data()
        self.bin_data()

        self.bin_mask()

        self.correct_mask_scaling()
        self.define_rebinned_mask()

        if self.do_plot:
            print(self.valid_mask.shape)
            io.plot(self.valid_mask[0].astype(np.float32), 'valid_mask[0]')

        self.prepare_stem_image()

        # self.valid_mask_nonzero_intensity = self.generate_valid_mask_nonzero_intensity()
        self.valid_mask_beam_valid = self.generate_valid_mask_beam_valid()

        # io.plot(valid_mask * data[0], 'test cropped valid mask 1')
        # io.plot(valid_mask_nonzero_intensity[0] * data[0], 'test cropped valid mask 2')

        self.maybe_interpolate_dead_pixels()

        if self.vacuum_measurements is not None:
            vacuum_data = self.data[self.vacuum_measurements]
            self.vacuum_mean = np.mean(vacuum_data, 0)
            vacuum_data = None
            if self.do_plot:
                io.plot(self.vacuum_mean, 'vacuum_mean')

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

        self.pos, self.pos1, self.pos2, self.pos3, self.pos4 = self.determine_positions()

        for i in range(self.pr.shape[0]):
            self.fpr[i] /= m.sqrt(norm(self.fpr[i]) ** 2)
            self.fpr[i] *= m.sqrt(self.beam_intensity)

            self.pr[i] /= m.sqrt(norm(self.pr[i]) ** 2)
            self.pr[i] *= m.sqrt(self.beam_intensity)

        if self.save_hdf5:
            fname = '%s%s_bin%d_%s.h5' % (self.save_path, self.name, self.binning_factor, self.save_suffix)
            print('Saving to hdf5 file %s ...' % fname)
            ret = self.prepare_hdf5_dict()
            rw.h5write(fname, ret)

        if self.save_matlab:
            fname = '%s%s_bin%d_%s' % (self.save_path, self.name, self.binning_factor, self.save_suffix)
            print('Saving to matlab file %s ...' % fname)
            from scipy.io import savemat
            ret = self.prepare_mat_dict()
            savemat(fname, ret, do_compression=True)
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        print(st)
        print('Done.')
        return ret
