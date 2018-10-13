from torch.nn.modules import Module
import skpr.nn.functional as F
from torch.autograd import Function, Variable
import skpr.inout as io

class AmplitudeLoss(Module):
    is_intensity_based = False

    def __init__(self, gradient_mask=None, beam_amplitude=1, a_h=5, M=[128, 128], NP=1, NO=1):
        super(AmplitudeLoss, self).__init__()
        self.a_h = a_h
        self.M = M
        self.NP = NP
        self.NO = NO
        self.gradient_mask = gradient_mask
        self.probe_amplitude = beam_amplitude

    def forward(self, I_model, a_target, mask):
        return F.amplitude_loss(I_model, a_target, mask)
