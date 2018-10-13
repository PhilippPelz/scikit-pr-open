import torch as th
from torch.autograd import Function, Variable

import skpr.inout as io
import skpr.nn as n


class CMul(Function):
    @staticmethod
    def probe_gradient_scaling(object_intensity, beta):
        # suggested 0.5 ... 5
        factor = (1 - beta) * object_intensity + beta * object_intensity.max()
        return factor.pow_(-1)

    @staticmethod
    def forward(ctx, P, O, dpos_proxy, q, beta):
        """
        Parameters
        ----------
        P           dimension: K, N_p, N_o, Mx, My
        O           dimension: K, N_p, N_o, Mx, My
        dpos_proxy  dimension: K, N_p, N_o, Mx, My
        shift_ramp  dimension: 2, Mx, My
        beta        dimension: 0

        Returns
        -------
        mul :           dimension: K, N_p, N_o, Mx, My      float tensor
        dpos_proxy :    dimension: 2, K, N_p, N_o, Mx, My   complex tensor
        """
        io.logger.debug('CMul forward 1')
        I_O = th.cuda.FloatTensor(*O.size())
        object_intensity = O.expect(out=I_O)
        object_intensity = th.sum(object_intensity, 2, keepdim=True)
        ctx.probe_grad_factor = CMul.probe_gradient_scaling(object_intensity, beta)
        ctx.two_pi_q = q
        ctx.cramp = dpos_proxy

        ctx.save_for_backward(P, O)
        io.logger.debug('CMul forward 2')
        mul = P * O
        cramp = dpos_proxy
        return mul, cramp

    @staticmethod
    def backward(ctx, grad_output, grad_dpos):
        """
        Parameters
        ----------
        grad_output : dimension: K, N_p, N_o, Mx, My
        grad_dpos : dimension: (2, K, NP, NO, Mx, My)   complex tensor

        Returns
        -------
        grad_P : dimension: K, N_p, N_o, Mx, My         float tensor
        grad_O : dimension: K, N_p, N_o, Mx, My         float tensor
        grad_dpos : dimension: (2, K, NP, NO, Mx, My)   complex tensor
        """
        probe_mode_dimension = 1
        object_mode_dimension = 2
        io.logger.debug('CMul backward 1')
        P, O = ctx.saved_tensors
        s = O.size()
        io.logger.debug('CMul backward 11')
        subpix_shift_ramp = ctx.cramp.conj()
        q_small = ctx.two_pi_q  # 2, Mx, My
        rs = q_small.size()
        two_pi_q = q_small.view(rs[0], 1, 1, 1, rs[1], rs[2]).expand(rs[0], s[0], s[1], s[2], rs[1], rs[2])

        d_drO = th.cuda.ZFloatTensor(2, *O.size())
        d_drO[0].copy_(O)
        d_drO[1].copy_(O)
        # n.var['d_drO_0'] = d_drO.cpu().numpy()[:, :, 0, 0, :, :]
        # print(subpix_shift_ramp.size())

        io.logger.debug('CMul backward 13')
        d_drO.fft2_()
        # n.var['d_drO_1'] = d_drO.cpu().numpy()[:, :, 0, 0, :, :]
        # n.var['q'] = q.cpu().numpy()[:, :, 0, 0, :, :]
        d_drO *= two_pi_q
        # n.var['subpix_shift_ramp'] = subpix_shift_ramp.cpu().numpy()[:, 0, 0, :, :]
        # n.var['d_drO_2'] = d_drO.cpu().numpy()[:, :, 0, 0, :, :]
        # d_drO *= subpix_shift_ramp
        # n.var['d_drO_3'] = d_drO.cpu().numpy()[:, :, 0, 0, :, :]
        d_drO.ifft2_()
        # n.var['d_drO_1'] = d_drO.cpu().numpy()[:, :, 0, 0, :, :]

        io.logger.debug('CMul backward 2')

        d_drOP = d_drO * P

        grad_dpos = d_drOP.conj_() * grad_dpos.data

        io.logger.debug('CMul backward 3')

        z_P = grad_output.data.new().resize_(*grad_output.data.size())
        z_O = grad_output.data.new().resize_(*grad_output.data.size())

        O.expand_as(z_P).conj(out=z_P)
        P.expand_as(z_O).conj(out=z_O)

        z_P *= grad_output.data  # probe gradient
        z_P *= ctx.probe_grad_factor.expand_as(z_P)
        grad_P = th.sum(z_P, object_mode_dimension, keepdim=True)

        z_O *= grad_output.data  # object gradient
        grad_O = th.sum(z_O, probe_mode_dimension, keepdim=True)

        io.logger.debug('CMul backward 4')
        return Variable(grad_P), Variable(grad_O), Variable(grad_dpos), None, None, None, None
