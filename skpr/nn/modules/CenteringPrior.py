# -*- coding: utf-8 -*
from torch.nn.modules import Module

import skpr.inout  as io
import skpr.nn.functional as F
from skpr.util import *


class CenteringPrior(Module):
    def __init__(self, NO, NP, M, radius_fraction, falloff_pixels):
        super(CenteringPrior, self).__init__()
        self.weight = th.from_numpy(-(1 - exp_decay_mask(M, M[0] / 2.0 * radius_fraction, falloff_pixels)))
        #        w = self.weight.numpy()
        #        io.plot(w, 'centering_weight')
        self.weight = self.weight.cuda().view(1, 1, M[0], M[1]).expand(NP, NO, M[0], M[1])

    def forward(self, target):
        io.logger.debug('CenteringPrior(Module) forward 1')
        # print 'BM3DPrior foward', type(target.data), type(self.alpha(epoch))
        loss = F.centering_prior(target.data, self.weight)
        io.logger.debug('CenteringPrior(Module) forward 2')
        return loss
