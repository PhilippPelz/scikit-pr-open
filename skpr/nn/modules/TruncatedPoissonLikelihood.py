from torch.nn.modules import Module
import skpr.nn.functional as F
from torch.autograd import Function, Variable


class TruncatedPoissonLikelihood(Module):
    is_intensity_based = True

    def __init__(self, gradient_mask=None, beam_amplitude=1, a_h=5, M=[128, 128], NP=1, NO=1):
        super(TruncatedPoissonLikelihood, self).__init__()
        self.a_h = a_h
        self.M = M
        self.NP = NP
        self.NO = NO
        self.gradient_mask = gradient_mask
        self.probe_amplitude = beam_amplitude

    def forward(self, I_model, I_target, mask):
        return F.truncated_poisson_likelihood(I_model, I_target, mask, self.a_h)
