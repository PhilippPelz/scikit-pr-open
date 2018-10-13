from torch.autograd import Function, Variable

import skpr.inout  as io


class IFFT(Function):
    @staticmethod
    def forward(ctx, input):
        io.logger.debug('FFT forward 1')
        return input.ifft2()

    @staticmethod
    def backward(ctx, grad_output):
        io.logger.debug('FFT backward 1')
        grad_input = grad_output.data.fft2()
        io.logger.debug('FFT backward 2')
        return Variable(grad_input), None, None
