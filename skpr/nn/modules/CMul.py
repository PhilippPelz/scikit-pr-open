import numpy as np
import torch as th
from numpy.fft import fftshift
from torch.nn.modules import Module

import skpr.nn.functional as F
from skpr import inout as io


class CMul(Module):
    def __init__(self, gradient_ramp, beta, epoch, subpixel_optimization_active=lambda epoch: False):
        super(CMul, self).__init__()
        self.subpixel_optimization_active = subpixel_optimization_active
        self.epoch = epoch
        self.gradient_ramp = gradient_ramp
        self.beta = beta

    def forward(self, P, O, dpos_proxy):
        if self.subpixel_optimization_active(self.epoch[0]):
            I = F.cmul(P, O, dpos_proxy, self.gradient_ramp, self.beta)
        else:
            I = F.cmul_no_subpixel_gradient(P, O, dpos_proxy, self.gradient_ramp, self.beta)
        return I
