import torch as th
from torch.autograd import Function
from torch.autograd import Variable

import skpr.inout  as io
from skpr._ext import skpr_thnn

denoise = skpr_thnn.denoise
complex2pair = skpr_thnn.CudaZFloatComplex2Pair
pair2complex = skpr_thnn.CudaZFloatPair2Complex


class CenteringPrior(Function):
    @staticmethod
    def forward(ctx, input, weight):
        io.logger.debug('CenteringPrior forward 1')
        # print input.size(), weight.size()
        ctx.prod = input * weight
        out = ctx.prod.norm().real
        io.logger.debug('CenteringPrior forward 2')
        return th.FloatTensor([out])

    @staticmethod
    def backward(ctx, grad_output):
        io.logger.debug('CenteringPrior backward 1')
        grad_input = ctx.prod
        #        x = grad_input.cpu().numpy()
        #        io.zplot([np.abs(x), np.angle(x)], 'CenteringPrior grad_input')
        io.logger.debug('CenteringPrior backward 2')
        return Variable(grad_input), None, None, None
