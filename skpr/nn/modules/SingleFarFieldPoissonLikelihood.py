# -*- coding: utf-8 -*-

from torch.nn.modules import Module

import skpr.nn.functional as F


class SingleFarFieldPoissonLikelihood(Module):
    is_intensity_based = True

    def __init__(self, gradient_mask=None, beam_amplitude=1, a_h=5, M=[128, 128], NP=1, NO=1):
        super(SingleFarFieldPoissonLikelihood, self).__init__()
        self.a_h = a_h
        self.M = M
        self.NP = NP
        self.NO = NO
        self.gradient_mask = gradient_mask
        self.probe_amplitude = beam_amplitude

    def forward(self, input, I_target, mask):
        return F.single_farfield_poisson_likelihood(input, I_target, self.probe_amplitude, mask, self.gradient_mask,
                                                    self.a_h,
                                                    self.M, self.NP, self.NO)
