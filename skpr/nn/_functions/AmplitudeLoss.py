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

class AmplitudeLoss(Function):
    @staticmethod
    def forward(ctx, I_model, a_target, valid_mask):
        """
        input       dimension: K, N_o, N_p, M1, M2
        I_target    dimension: K, M1, M2
        valid_mask        dimension: K, M1, M2

                Should be overridden by all subclasses.
        """
        # print 'FarFieldPoissonLikelihood.forward'
        if 'Float' in type(I_model).__name__:
            likelihood = skpr_thnn.CudaZFloatEuclideanLoss_updateOutput
        elif 'Double' in type(I_model).__name__:
            likelihood = skpr_thnn.CudaZDoubleEuclideanLoss_updateOutput
        else:
            raise NotImplementedError()

        a_model = I_model.sqrt_()
        # x = a_model.cpu().numpy()
        # io.plot_abs_phase_mosaic(x)

        ctx.save_for_backward(a_target, valid_mask)
        ctx.a_model = a_model

        if p.var['i'] % p.var['period'] == 0 and p.var['i'] > 0:
            pass
            mask = fftshift(valid_mask.cpu().numpy(), (1, 2))
            I = fftshift(a_model.squeeze().cpu().numpy(), (1, 2))
            It = fftshift(a_target.cpu().numpy(), (1, 2))
            io.plotzmosaic([I, It * mask], 'a_model vs a_target', cmap=['inferno', 'inferno'], title=['I', 'It'])

        out = th.cuda.FloatTensor(1)
        likelihood(a_model, a_target, valid_mask, out)
        io.logger.debug('AmplitudeLoss out = %3.2g' % out[0])
        return out.cpu()

    @staticmethod
    def backward(ctx, grad_output):
        io.logger.debug('AmplitudeLoss backward 1')
        if 'Float' in type(ctx.a_model).__name__:
            gradient_factor = skpr_thnn.CudaZFloatEuclideanLoss_GradientFactor
        elif 'Double' in type(ctx.a_model).__name__:
            gradient_factor = skpr_thnn.CudaZFloatEuclideanLoss_GradientFactor
        else:
            raise NotImplementedError()

        a_target, valid_mask = ctx.saved_tensors
        gf = ctx.a_model.clone()

        gfc = fftshift(gf.cpu().numpy(), (1, 2))
        n.var['gfc1'] = gfc

        gradient_factor(gf, a_target, valid_mask)

        gfc = fftshift(gf.cpu().numpy(), (1, 2))
        n.var['gfc2'] = gfc


        io.logger.debug('AmplitudeLoss backward 3')

        grad_input = Variable(gf)
        return grad_input, None, None, None
