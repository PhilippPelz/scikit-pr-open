from torch.autograd import Function, Variable

import skpr.inout  as io


class FFTShift(Function):
    @staticmethod
    def forward(ctx, input, axes):
        io.logger.debug('FFTShift forward 1')
        ctx.axes = axes
        return input.fftshift(axes)

    @staticmethod
    def backward(ctx, grad_output):
        io.logger.debug('FFTShift backward 1')
        grad_input = grad_output.data.ifftshift(ctx.axes)
        io.logger.debug('FFTShift backward 2')
        return Variable(grad_input), None
