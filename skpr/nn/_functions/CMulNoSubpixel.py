import torch as th
from torch.autograd import Function, Variable

import skpr.inout  as io
import numpy as np
import skpr.util as u
import skpr.nn as n


class CMulNoSubpixel(Function):
    @staticmethod
    def probe_gradient_scaling(object_intensity, beta):
        factor = (1 - beta) * object_intensity + beta * object_intensity.max()
        return factor.pow_(-1)

    @staticmethod
    def forward(ctx, P, O, dpos_proxy, shift_ramp, beta):
        """
        x           dimension: K, N_p, N_o, M1, M2
        y           dimension: K, N_p, N_o, M1, M2
        dpos_proxy  dimension: 2, K, N_p, N_o, M1, M2
        K           dimension: 0
        """
        io.logger.debug('CMulNoSubpixel forward 1')
        I_O = th.cuda.FloatTensor(*O.size())
        object_intensity = O.expect(out=I_O)
        object_intensity = th.sum(object_intensity, 2, keepdim=True)
        ctx.probe_grad_factor = CMulNoSubpixel.probe_gradient_scaling(object_intensity, beta)
        ctx.shift_ramp = shift_ramp * -1

        ctx.save_for_backward(P, O)
        io.logger.debug('CMulNoSubpixel forward 2')
        return P * O, dpos_proxy

    @staticmethod
    def backward(ctx, grad_output, grad_dpos):
        probe_mode_dimension = 1
        object_mode_dimension = 2
        io.logger.debug('CMulNoSubpixel backward 1')
        P, O = ctx.saved_tensors

        z_P = grad_output.data.new().resize_(*grad_output.data.size())
        z_O = grad_output.data.new().resize_(*grad_output.data.size())

        O.expand_as(z_P).conj(out=z_P)
        P.expand_as(z_O).conj(out=z_O)

        z_P *= grad_output.data  # probe gradient
        z_P *= ctx.probe_grad_factor.expand_as(z_P)
        grad_P = th.sum(z_P, object_mode_dimension, keepdim=True)

        z_O *= grad_output.data  # object gradient
        grad_O = th.sum(z_O, probe_mode_dimension, keepdim=True)
        # n.var['grad_O'] = grad_O.cpu().numpy()[ :, 0, 0, :, :]

        io.logger.debug('CMulNoSubpixel backward 2')
        return Variable(grad_P), Variable(grad_O), None, None, None
