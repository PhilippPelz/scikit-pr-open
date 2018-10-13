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

class TruncatedPoissonLikelihood(Function):
    @staticmethod
    def forward(ctx, I_model, I_target, valid_mask, a_h):
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
        ctx.a_h = a_h
        ctx.save_for_backward(I_target, valid_mask)
        ctx.I_model = I_model

        if p.var['i'] % p.var['period'] == 0 and p.var['i'] > 0:
            pass
            # mask = fftshift(valid_mask.cpu().numpy(), (1, 2))
            # I = fftshift(I_model.squeeze().cpu().numpy(), (1, 2))
            # It = fftshift(I_target.cpu().numpy(), (1, 2))
            # io.plotzmosaic([I, It * mask], 'I_model vs I_target', cmap=['inferno', 'inferno'], title=['I', 'It'])

        out = th.cuda.FloatTensor(1)
        likelihood(I_model, I_target, valid_mask, out)
        io.logger.debug('TruncatedPoissonLikelihood out = %3.2g' % out[0])
        return out.cpu()

    @staticmethod
    def backward(ctx, grad_output):
        io.logger.debug('TruncatedPoissonLikelihood backward 1')
        if 'Float' in type(ctx.I_model).__name__:
            gradient_factor = skpr_thnn.CudaZFloatTruncatedPoissonLikelihood_GradientFactor
        elif 'Double' in type(ctx.I_model).__name__:
            gradient_factor = skpr_thnn.CudaZFloatTruncatedPoissonLikelihood_GradientFactor
        else:
            raise NotImplementedError()

        I_target, valid_mask = ctx.saved_tensors
        gf = ctx.I_model.clone()

        K_t = (gf - I_target).norm(1)/I_target.numel()
        lhs = (gf - I_target).abs()
        truncation_mask = (lhs <= K_t * ctx.a_h).float()

        GS = np.array(I_target.size())
        for dim in range(1, I_target.ndimension() - 2):
            GS[dim] = 1
        GS = th.Size(GS)
        truncation_mask = truncation_mask.view(GS).expand_as(I_target)

        gradient_factor(gf, I_target, valid_mask)

        # gfc = fftshift(gf.cpu().numpy(), (1, 2))
        # n.var['gfc'] = gfc
        gf *= truncation_mask

        io.logger.debug('TruncatedPoissonLikelihood backward 3')

        grad_input = Variable(gf)
        return grad_input, None, None, None
