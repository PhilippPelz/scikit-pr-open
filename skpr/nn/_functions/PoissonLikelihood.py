# -*- coding: utf-8 -*-

import numpy as np
import torch as th
from numpy.fft import fftshift
from torch.autograd import Function, Variable

import skpr.inout  as io
import skpr.nn as p
from skpr._ext import skpr_thnn
import skpr.nn as n

# void CudaZFloatTruncatedPoissonLikelihood_updateOutput(THCudaTensor *input, THCudaTensor *target, THCudaTensor *valid_mask, THCudaTensor *output);
# void CudaZFloatTruncatedPoissonLikelihood_GradientFactor(THCudaTensor *input, THCudaTensor *target, THCudaTensor *valid_mask);
# void CudaZDoubleTruncatedPoissonLikelihood_updateOutput(THCudaDoubleTensor *input, THCudaDoubleTensor *target, THCudaDoubleTensor *valid_mask, THCudaDoubleTensor *output);
# void CudaZDoubleTruncatedPoissonLikelihood_GradientFactor(THCudaDoubleTensor *input, THCudaDoubleTensor *target, THCudaDoubleTensor *valid_mask);

class PoissonLikelihood(Function):
    @staticmethod
    def forward(ctx, I_model, I_target, valid_mask):
        """
        input       dimension: K, N_o, N_p, M1, M2
        I_target    dimension: K, M1, M2
        valid_mask        dimension: K, M1, M2

                Should be overridden by all subclasses.
        """
        # print 'FarFieldPoissonLikelihood.forward'
        if 'Float' in type(I_model).__name__:
            likelihood = skpr_thnn.CudaZFloatTruncatedPoissonLikelihood_updateOutput
        elif 'Double' in type(I_model).__name__:
            likelihood = skpr_thnn.CudaZDoubleTruncatedPoissonLikelihood_updateOutput
        else:
            raise NotImplementedError()

        ctx.save_for_backward(I_target, valid_mask)
        ctx.I_model = I_model

        if p.var['i'] % p.var['period'] == 0 and p.var['i'] > 0:
            pass
            mask = fftshift(valid_mask.cpu().numpy(), (1, 2))
            I = fftshift(I_model.squeeze().cpu().numpy(), (1, 2))
            It = fftshift(I_target.cpu().numpy(), (1, 2))
            io.plotzmosaic([I, It * mask], 'I_model vs I_target', cmap=['inferno', 'inferno'], title=['I', 'It'])

        out = th.cuda.FloatTensor(1)
        likelihood(I_model, I_target, valid_mask, out)
        io.logger.debug('PoissonLikelihood out = %3.2g' % out[0])
        return out.cpu()

    @staticmethod
    def backward(ctx, grad_output):
        io.logger.debug('PoissonLikelihood backward 1')
        if 'Float' in type(ctx.I_model).__name__:
            gradient_factor = skpr_thnn.CudaZFloatTruncatedPoissonLikelihood_GradientFactor
        elif 'Double' in type(ctx.I_model).__name__:
            gradient_factor = skpr_thnn.CudaZFloatTruncatedPoissonLikelihood_GradientFactor
        else:
            raise NotImplementedError()

        I_target, valid_mask = ctx.saved_tensors
        gf = ctx.I_model.clone()

        gradient_factor(gf, I_target, valid_mask)

        # gfc = fftshift(gf.cpu().numpy(), (1, 2))
        # n.var['gfc1'] = gfc

        io.logger.debug('PoissonLikelihood backward 3')

        grad_input = Variable(gf)
        return grad_input, None, None, None
