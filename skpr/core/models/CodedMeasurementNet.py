#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 15:13:53 2017

@author: pelzphil
"""

import torch as th
import numpy as np
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import Module
from skpr.inout.plot import plot, zplot
import skpr.nn.functional as F
import skpr.nn.modules  as M
import skpr.inout  as io
from . import *


class CodedMeasurementNet(Module):
    def __init__(self, K, M, N, m_init=None, parallel_type='none'):
        super(CodedMeasurementNet, self).__init__()
        self.K = K
        random_vars = th.rand(K, *N)
        if m_init is not None:
            self.m = m_init
        else:
            self.m = ((random_vars < 0.5).float() * 2 - 1)
            self.m = self.m.float().cuda()

    def forward(self, i):
        io.logger.debug('CodedMeasurementNet forward 1')
        in_broadcast = F.broadcast(i, self.K, False)
        # print type(in_broadcast)
        # print type(self.m)
        ii  = i.data.cpu().numpy()
        io.zplot([np.real(ii),np.imag(ii)],'in')
        out = F.cmul(in_broadcast.data, self.m)
        ii  = out.data.cpu().numpy()
        io.zplot([np.real(ii[0]),np.imag(ii[0])],'out')
        io.logger.debug('CodedMeasurementNet forward 2')
        return out
