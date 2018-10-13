import numpy as np
import torch as th
from torch.autograd import Function, Variable

import skpr.inout as io


class SubpixelShiftNoShiftGrad(Function):
    @staticmethod
    def forward(ctx, input, shift, ramps0):
        """
        input               dimension: K, N_o, N_p, M1, M2
        shift               dimension: K, 2
        ramps0              dimension: 2, M1, M2
        """
        io.logger.debug('SubpixelShiftNoShiftGrad forward 1')
        s = np.array(input.size())

        shx = shift[:, 0].contiguous()  # K, 1
        shy = shift[:, 1].contiguous()  # K, 1

        # print('shifts')
        # print(shx)
        # print(shy)

        # K, 1 --> K, Mx, My
        shx = shx.view(s[0], 1, 1).expand(s[0], s[-2], s[-1])
        shy = shy.view(s[0], 1, 1).expand(s[0], s[-2], s[-1])

        rs = ramps0.size()
        # print 'rs', rs
        ramps = ramps0.clone()
        # io.plot(ramps[0].cpu().numpy(),'ramp x')
        # io.plot(ramps[1].cpu().numpy(), 'ramp y')
        # print 'ramps size: %f MB' % 2 ** (np.log2(ramps.nelement() * 4) - 20)

        # 2, Mx, My --> 2, K, Mx, My
        ramps = ramps.view(rs[0], 1, rs[1], rs[2]).repeat(1, s[0], 1, 1)
        # print 'ramps.size ', ramps.size()
        # print 'ramps size: %f MB' % 2 ** (np.log2(ramps.nelement() * 4) - 20)

        #        print ramps[0].size(), shx.size()
        ramps[0] *= shx
        ramps[1] *= shy
        # io.plotmosaic(ramps[0].cpu().numpy(), 'ramps x')
        # io.plotmosaic(ramps[1].cpu().numpy(), 'ramps y')

        ctx.ramp = ramps.clone().view(2, s[0], 1, 1, rs[1], rs[2]).expand(2, s[0], s[1], s[2], rs[1], rs[2])
        # print 'ctx.ramp size: %f MB' % 2 ** (np.log2(ctx.ramp.nelement() * 4) - 20)
        ctx.rs = rs
        ctx.s = s
        #        io.plot(ramps[0,0].cpu().numpy(), 'ramp0')
        #        io.plot(ramps[1,0].cpu().numpy(), 'ramp1')

        r = th.sum(ramps, 0)

        # K, Mx, My --> K, 1, 1, Mx, My
        ramp = r.view(s[0], 1, 1, rs[1], rs[2])  # .expand(rs[0], s[1], s[2], rs[1], rs[2])
        # x = ramp.cpu().numpy()[:, 0, 0, :, :]
        # io.plot_abs_phase_mosaic(x, 'ramp')

        is_cuda = 'cuda' in str(type(input))
        c = th.cuda if is_cuda else th
        ctx.c = c
        cramps = c.ZFloatTensor(s[0], 1, 1, rs[1], rs[2])
        # print 'cramps size: %f MB' % 2 ** (np.log2(ctx.ramp.nelement() * 8) - 20)
        cramps.polar_(1.0, ramp)
        cramps = cramps.expand_as(input)
        ctx.cramp = cramps

        out = input.clone()

        out.fft2_()
        out *= cramps
        out.ifft2_()

        io.logger.debug('SubpixelShiftNoShiftGrad forward 2')
        return out, shift

    @staticmethod
    def backward(ctx, grad_output, grad_shift):
        """
        grad_output             dimension: K, N_o, N_p, M1, M2
        grad_input              dimension: K, N_o, N_p, M1, M2
        grad_shift              dimension: K, 2
        """
        io.logger.debug('SubpixelShiftNoShiftGrad backward 1')

        cramp_conj = ctx.cramp.conj_()
        grad_input = grad_output.data.clone()
        # p.var['grad_input.in'] = grad_input.cpu().numpy()[:, 0, 0, :, :]
        grad_input.fft2_()
        grad_input *= cramp_conj
        grad_input.ifft2_()
        # p.var['grad_input.ifft2'] = grad_input.cpu().numpy()[:, 0, 0, :, :]

        io.logger.debug('SubpixelShiftNoShiftGrad backward 2')
        return Variable(grad_input), None, None
