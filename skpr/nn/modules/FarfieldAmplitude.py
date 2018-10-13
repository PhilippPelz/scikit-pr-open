import numpy as np
import torch as th
from numpy.fft import fftshift
from torch.nn.modules import Module

import skpr.nn.functional as F
from skpr import inout as io


class FarfieldAmplitude(Module):
    def __init__(self, epoch, gradient_mask=None, subpixel_optimization_active=lambda epoch: False):
        super(FarfieldAmplitude, self).__init__()
        self.subpixel_optimization_active = subpixel_optimization_active
        self.epoch = epoch
        self.gradient_mask = gradient_mask

    def forward(self, exit_waves, dpos):
        if self.subpixel_optimization_active(self.epoch[0]):
            I = F.farfield_amplitude(exit_waves, dpos, self.gradient_mask)
        else:
            I = F.farfield_amplitude_no_subpixel_gradient(exit_waves, self.gradient_mask)
        return I
