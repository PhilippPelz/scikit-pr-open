import torch as th
from torch.autograd import Function

import skpr.inout  as io


class Broadcast(Function):
    @staticmethod
    def forward(ctx, input, ntimes, divide_by_ntimes=True):
        io.logger.debug('Broadcast forward 1')
        # out = input.clone()
        # print 'Broadcast.forward'
        ctx.ntimes = ntimes
        ctx.divide_by_ntimes = divide_by_ntimes
        io.logger.debug('Broadcast forward 2')
        return input.view(1, *input.size()).expand(ntimes, *input.size())

    @staticmethod
    def backward(ctx, grad_output):
        io.logger.debug('Broadcast backward 1')
        # p.var['dCMul'] = grad_output.data.cpu().squeeze().numpy()
        grad_input = th.sum(grad_output, 0)
        if ctx.divide_by_ntimes:
            grad_input.data.div_(ctx.ntimes)
        # p.var['dP'] = grad_input.data.cpu().numpy()
        io.logger.debug('Broadcast backward 2')
        return grad_input, None, None
