import numpy as np
import torch as th
from torch.autograd import Function
from torch.autograd import Variable

import skpr.inout  as io
import skpr.nn as p
from skpr._ext import skpr_thnn

denoise = skpr_thnn.denoise
complex2pair = skpr_thnn.CudaZFloatComplex2Pair
pair2complex = skpr_thnn.CudaZFloatPair2Complex


class BM3DPrior(Function):
    @staticmethod
    def forward(ctx, input, sigma):
        io.logger.debug('BM3DPrior forward 1')

        # in: NP, NO, Nx, Ny

        stackdim = 1
        in_size = input.size()
        size = np.array(in_size)
        # print size,size[:-2]
        for s in size[:-2]:
            stackdim *= s
        view_size = np.array((stackdim, size[-2], size[-1]))
        # view_size: NP * NO, Nx, Ny

        input_3dstack = input.view(*view_size)

        pair_size = np.copy(view_size)
        pair_size[0] *= 2
        # pair_size: 2 * NP * NO, Nx, Ny

        pair = th.cuda.FloatTensor(2, *input_3dstack.size())
        # pair: 2, NP * NO, Nx, Ny
        pair_denoise_view = pair.view(*pair_size)
        # pair_denoise_view: 2 * NP * NO, Nx, Ny
        denoised_bytes = th.cuda.ByteTensor(*pair_size)
        denoised = th.cuda.ZFloatTensor(*input_3dstack.size())

        complex2pair(input_3dstack, pair)
        th.cuda.synchronize()

        mins = [1e-10, 1e-10]
        maxs = [1e-10, 1e-10]

        mins[0] = pair[0].min()
        mins[1] = pair[1].min()
        # print 'mins ', mins

        pair[0].sub_(mins[0])
        pair[1].sub_(mins[1])

        maxs[0] = pair[0].max()
        maxs[1] = pair[1].max()

        # print 'maxs ', maxs

        pair[0].div_(maxs[0])
        pair[1].div_(maxs[1])

        pair *= 256

        # print pair[0].squeeze().size()
        # print type(pair[0])
        # io.showPlot(pair[0].squeeze().cpu().numpy(), True)
        # io.showPlot(pair[1].byte().float().squeeze().cpu().numpy(), True)

        in_byte = pair_denoise_view.byte()

        # print in_byte.size(), denoised_bytes.size()
        # print type(in_byte), type(denoised_bytes)
        denoise(in_byte, denoised_bytes, th.FloatTensor([sigma]))
        th.cuda.synchronize()
        # io.showPlot(denoised_bytes.cpu().float().numpy()[0], True,title='denoised_bytes')

        denoised_bytes_pair_view = denoised_bytes.view(2, *input_3dstack.size())
        # denoised_bytes_pair_view: 2, NP * NO, Nx, Ny

        # im = denoised_bytes_pair_view.cpu().float().squeeze().numpy()
        # io.zplot([im[0], im[1]], 'denoised')

        denoised_pair = denoised_bytes_pair_view.float() / 256

        denoised_pair[0].mul_(maxs[0])
        denoised_pair[1].mul_(maxs[1])

        denoised_pair[0].add_(mins[0])
        denoised_pair[1].add_(mins[1])

        # io.showPlot(denoised_pair.cpu().float().numpy()[0,0], True, title='denoised_pair')

        # im = denoised_pair.cpu().squeeze().numpy()
        # io.zplot([im[0], im[1]], 'denoised_pair')
        pair2complex(denoised_pair, denoised)
        th.cuda.synchronize()

        # im = denoised.cpu().squeeze().numpy()
        # io.zplot([np.real(im), np.imag(im)], 'denoised')

        denoised = denoised.view(*in_size)
        # p.var['denoised'] = denoised.cpu().numpy()
        ctx.diff = input - denoised
        # p.var['diff'] = ctx.diff[0, 0].cpu().numpy()
        # im = ctx.diff.cpu().squeeze().numpy()
        # io.zplot([np.real(im), np.imag(im)], 'diff denoised')

        out = th.FloatTensor(1)
        out.fill_(ctx.diff.norm().real)
        io.logger.debug('BM3DPrior forward 2')
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors
        io.logger.debug('BM3DPrior backward 1')
        grad_input = ctx.diff
        p.var['denoised_grad'] = grad_input.cpu().numpy()[0, 0]
        io.logger.debug('BM3DPrior backward 2')
        return Variable(grad_input), None, None, None
