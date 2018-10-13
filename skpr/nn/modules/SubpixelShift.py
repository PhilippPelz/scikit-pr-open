import numpy as np
import torch as th
from numpy.fft import fftshift
from torch.nn.modules import Module

import skpr.nn.functional as F
from skpr import inout as io
import skpr.util as u

class SubpixelShift(Module):
    def __init__(self, M, shift_ramp=None, epoch=np.array(0), subpixel_optimization_active=lambda epoch: False):
        super(SubpixelShift, self).__init__()
        self.subpixel_optimization_active = subpixel_optimization_active
        self.epoch = epoch

        if shift_ramp is not None:
            self.ramp = shift_ramp
        else:
            q = u.get_shift_ramp(M)
            self.ramp = th.from_numpy(- 2 * np.pi *q).cuda()

            # print 'ramp size: %f MB' % 2 ** (np.log2(self.ramp.nelement() * 4) - 20)

            # io.logger.debug('ramp size: %f MB' % 2 ** (np.log2(self.ramp.nelement() * 4) - 20))

    def forward(self, target, pos):
        """
        Forward.
        Parameters
        ----------
        target : (Nbatch, NP, NO, Mx, My) complex tensor
        pos : (Nbatch, 2) float tensor, positions (subpixel float part)
        Returns
        -------
        shifted : (Nbatch, NP, NO, Mx, My) complex tensor
            shifted waves at positions pos.
        """
        # x = target.data.cpu().numpy()[:,0,0,:,:]
        # print x.shape
        # io.plot_abs_phase_mosaic(x,'waves_unshifted')
        #        io.logger.debug('SubpixelShift(Module) forward 1')
        if self.subpixel_optimization_active(self.epoch[0]):
            shifted, pos_proxy = F.subpixel_shift(target, pos, self.ramp)
        else:
            shifted, pos_proxy = F.subpixel_shift_no_shift_gradients(target, pos, self.ramp)
        # x = shifted.data.cpu().numpy()[:, 0, 0, :, :]
        # io.plot_abs_phase_mosaic(x, 'waves_ shifted')
        # io.logger.debug('SubpixelShift(Module) forward 2')
        return shifted, pos_proxy
