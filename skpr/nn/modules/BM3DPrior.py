# -*- coding: utf-8 -*
import torch as th
from torch.nn.modules import Module
import skpr.nn.functional as F
from torch.autograd import Function, Variable
import skpr.inout  as io


class BM3DPrior(Module):
    def __init__(self, sigma):
        super(BM3DPrior, self).__init__()
        self.sigma = sigma

    def forward(self, target, epoch):
        io.logger.debug('BM3DPrior(Module) forward 1')
        # print 'BM3DPrior foward', type(target.data), type(self.alpha(epoch))
        loss = F.bm3d_prior(target, self.sigma)
        io.logger.debug('BM3DPrior(Module) forward 2')
        return loss
