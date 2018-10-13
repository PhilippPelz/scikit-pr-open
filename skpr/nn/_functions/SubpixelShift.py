import numpy as np
import torch as th
from torch.autograd import Function, Variable

import skpr.inout as io
import skpr.nn as p

class SubpixelShift(Function):
    @staticmethod
    def forward(ctx, input, shift, ramps0):
        """
        input               dimension: K, N_o, N_p, M1, M2
        shift               dimension: K, 2
        ramps0              dimension: 2, M1, M2

        Returns
        -------
        out : dimension: K, N_o, N_p, M1, M2 complex, shifted input
        cramp: dimension: 2, K, N_o, N_p, M1, M2 complex, dummy variable
        """
        io.logger.debug('SubpixelShift forward 1')
        s = np.array(input.size())

        #        print 'shift :', shift.cpu().numpy()

        shx = shift[:, 0].contiguous()  # K, 1
        shy = shift[:, 1].contiguous()  # K, 1

        # K, 1 --> K, Mx, My
        shx = shx.view(s[0], 1, 1).expand(s[0], s[-2], s[-1])
        shy = shy.view(s[0], 1, 1).expand(s[0], s[-2], s[-1])

        rs = ramps0.size()

        ramps = ramps0.clone()

        # 2, Mx, My --> 2, K, Mx, My
        ramps = ramps.view(rs[0], 1, rs[1], rs[2]).repeat(1, s[0], 1, 1)
        # print 'ramps size: %f MB' % 2 ** (np.log2(ramps.nelement() * 4) - 20)

        #        print ramps[0].size(), shx.size()
        ramps[0] *= shx
        ramps[1] *= shy

        ctx.ramp = ramps.clone().view(2, s[0], 1, 1, rs[1], rs[2]).expand(2, s[0], s[1], s[2], rs[1], rs[2])
        # print 'ctx.ramp size: %f MB' % 2 ** (np.log2(ctx.ramp.nelement() * 4) - 20)
        ctx.rs = rs
        ctx.s = s

        #        io.plot(ramps[0,0].cpu().numpy(), 'ramp0')
        #        io.plot(ramps[1,0].cpu().numpy(), 'ramp1')

        r = th.sum(ramps, 0)

        # K, Mx, My --> K, 1, 1, Mx, My
        ramp = r.view(s[0], 1, 1, rs[1], rs[2])  # .expand(rs[0], s[1], s[2], rs[1], rs[2])
        # print 'ramp size: %f MB' % 2 ** (np.log2(ramp.nelement() * 4) - 20)

        is_cuda = 'cuda' in str(type(input))
        c = th.cuda if is_cuda else th
        ctx.c = c
        cramps = c.ZFloatTensor(s[0], 1, 1, rs[1], rs[2])
        cramps.polar_(1.0, ramp)
        cramps = cramps.expand_as(input)
        # print 'cramps size: %f MB' % 2 ** (np.log2(ctx.ramp.nelement() * 8) - 20)
        ctx.cramp = cramps

        # x = input.cpu().numpy()[:,0,0,:,:]
        # io.plot_abs_phase_mosaic(x, 'input')

        out = input.clone()
        out.fft2_()

        # x = out.cpu().numpy()[:, 0, 0, :, :]
        # io.plot_abs_phase_mosaic(x, 'fft(input)')

        out *= cramps

        # x = out.cpu().numpy()[:, 0, 0, :, :]
        # io.plot_abs_phase_mosaic(x, 'out *= cramps')

        out.ifft2_()

        # x = out.cpu().numpy()[:, 0, 0, :, :]
        # io.plot_abs_phase_mosaic(x, 'out.ifft2_()')
        io.logger.debug('SubpixelShift forward 2')
        return out, ctx.cramp

    @staticmethod
    def backward(ctx, grad_output, grad_shift):
        """
        grad_output             dimension: K, N_o, N_p, M1, M2
        grad_input              dimension: K, N_o, N_p, M1, M2
        grad_shift              dimension: K, 2

        grad_output             not shifted
        grad_input.fft2_()          shifted
        grad_input final        not shifted

        grad_ramp                   shifted
        ramp                        shifted
        grad_shifts_all             shifted
        """
        io.logger.debug('SubpixelShift backward 1')
        grad_input = grad_output.data.clone()
        cramp_conj = ctx.cramp.conj_()
        # p.var['grad_input.in'] = in_fft.cpu().numpy()[:, 0, 0, :, :]
        # p.var['grad_output'] = grad_input.cpu().numpy()[:, 0, 0, :, :]
        # grad_input = grad_input.fftshift(axes=(3, 4))
        grad_input.fft2_()
        # grad_input = grad_input.fftshift(axes=(3, 4))
        # p.var['grad_input.fft2'] = grad_input.cpu().numpy()[:, 0, 0, :, :]

        grad_input *= cramp_conj
        grad_input.ifft2_()

        im = ctx.c.FloatTensor(grad_shift.size())
        grad_shift.data.re(out=im)  # dimension: 2, K, N_o, N_p, M1, M2
        for dim in [2, 3, 4, 5]:
            im = th.sum(im, dim, keepdim=True)
        # print 'im size' , im.size()
        for dim in [2, 2, 2, 2]:
            im = im.squeeze(dim)  # dimension: 2, K
        # print type(im)
        io.logger.debug('SubpixelShift backward 2')
        grad_shift = im.t()
        # io.logger.debug('grad_shift.max() = %g ' % grad_shift.max())
        return Variable(grad_input), Variable(grad_shift), None  # Variable(grad_shift), None
