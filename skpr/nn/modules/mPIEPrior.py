# -*- coding: utf-8 -*-
from astropy.visualization import hist
from torch.nn.modules import Module
import skpr.nn.functional as F
from torch.autograd import Function, Variable


class mPIEPrior(Module):
    is_intensity_based = True

    def __init__(self, alpha, history_size):
        super(mPIEPrior, self).__init__()
        self.alpha = alpha
        self.history_size = history_size
        self.input_history = None

    def forward(self, input, epoch):
        if self.input_history is None:
            self.input_history = th.cuda.ZFloatTensor()
        if self.last_input is not None:
            loss = F.rpie_prior(input, self.alpha, self.last_input, probe)
        else:
            loss = 0
        self.last_input = input
        return loss

