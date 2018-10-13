# -*- coding: utf-8 -*-

import numpy as np
import torch as th
from numpy.fft import fftshift
from torch.autograd import Function, Variable

# void CudaZFloatTruncatedPoissonLikelihood_updateOutput(THCudaTensor *input, THCudaTensor *target, THCudaTensor *valid_mask, THCudaTensor *output);
# void CudaZFloatTruncatedPoissonLikelihood_GradientFactor(THCudaTensor *input, THCudaTensor *target, THCudaTensor *valid_mask);
# void CudaZDoubleTruncatedPoissonLikelihood_updateOutput(THCudaDoubleTensor *input, THCudaDoubleTensor *target, THCudaDoubleTensor *valid_mask, THCudaDoubleTensor *output);
# void CudaZDoubleTruncatedPoissonLikelihood_GradientFactor(THCudaDoubleTensor *input, THCudaDoubleTensor *target, THCudaDoubleTensor *valid_mask);
import skpr.inout  as io
import skpr.nn as p
from skpr._ext import skpr_thnn


class FarFieldPoissonLikelihood(Function):
    i = 0

    @staticmethod
    def forward(ctx, input, dpos, I_target, beam_amplitude, valid_mask, gradient_mask, a_h, M, NP, NO):
        """
        input       dimension: K, N_o, N_p, M1, M2
        I_target    dimension: K, M1, M2
        valid_mask        dimension: K, M1, M2

                Should be overridden by all subclasses.
        """
        # print 'FarFieldPoissonLikelihood.forward'
        if 'ZFloat' in type(input).__name__:
            likelihood = skpr_thnn.CudaZFloatTruncatedPoissonLikelihood_updateOutput
        elif 'ZDouble' in type(input).__name__:
            likelihood = skpr_thnn.CudaZDoubleTruncatedPoissonLikelihood_updateOutput
        else:
            raise NotImplementedError()
        # print input.norm()
        # e1 = input.cpu().numpy()[0, 0, 0]
        # io.zplot([np.abs(e1), np.angle(e1)], 'input')
        in_shifted = input.ifftshift((3, 4))
        PsiF_model = in_shifted.fft2_()
        #        print PsiF_model.norm()
        ctx.PsiF_model = PsiF_model
        ctx.gradient_mask = gradient_mask
        ctx.a_h = a_h
        ctx.M = M
        ctx.NP = NP
        ctx.NO = NO
        ctx.probe_amplitude = beam_amplitude

        # PsiF_model.mul_(beam_amplitude[0])
        I_model = th.cuda.FloatTensor(input.size())
        # print 'I_model size', I_model.size()
        #        print 'I_model',dir(I_model)
        #        print dir(PsiF_model)
        PsiF_model.expect(out=I_model)

        #        print(I.shape)
        #        print(It.shape)
        #        plot(np.log10(I[0,0,0]))
        #        plot(np.log10(It[0]))
        #         print np.sum(I[633,0,0]), np.sum(It[633])
        # zplot([np.log10(I),np.log10(It)*mask],'model  - target', cmap=['hot','hot'])
        #        zplot([np.log10(I[1,0,0]),np.log10(It[1])],'model  - target 1', cmap=['hot','hot'])
        #        zplot([np.log10(I[2,0,0]),np.log10(It[2])],'model  - target 2', cmap=['hot','hot'])
        #        zplot([np.log10(I[3,0,0]),np.log10(It[3])],'model  - target 3', cmap=['hot','hot'])
        #        quit()
        # sum up all dimensions but the one with indices 0 (batch dimension) and size-1, size-2 (image dimensions)
        for dim in range(1, I_model.ndimension() - 2):
            I_model = th.sum(I_model, dim, keepdim=True)

        ctx.save_for_backward(I_target, valid_mask)
        I_model = I_model.squeeze()
        ctx.I_model = I_model

        if p.var['i'] % p.var['period'] == 0 and p.var['i'] > 0:
            pass
            # psi = PsiF_model.cpu().numpy()[0,0,0]
            # io.zplot([np.abs(psi),np.angle(psi)],'psi_fourier')
            # mask = fftshift(valid_mask.cpu().numpy(), (1, 2))
            # I = fftshift(I_model.squeeze().cpu().numpy(), (1, 2))
            # It = fftshift(I_target.cpu().numpy(), (1, 2))
            # io.plotzmosaic([I, It * mask], 'I_model vs I_target', cmap=['inferno', 'inferno'], title=['I', 'It'])

        out = th.cuda.FloatTensor(1)
        likelihood(I_model, I_target, valid_mask, out)
        io.logger.debug('FarFieldPoissonLikelihood out = %3.2g' % out[0])
        return out.cpu()

    @staticmethod
    def backward(ctx, grad_output):
        io.logger.debug('FarFieldPoissonLikelihood backward 1')
        PsiF_model = ctx.PsiF_model
        a_h = ctx.a_h
        grad_input = PsiF_model.clone()

        # p.var['grad_input'] = fftshift(x[:, 0, 0, :, :], axes=(1, 2))
        if 'ZFloat' in type(PsiF_model).__name__:
            gradient_factor = skpr_thnn.CudaZFloatTruncatedPoissonLikelihood_GradientFactor
        elif 'ZDouble' in type(PsiF_model).__name__:
            gradient_factor = skpr_thnn.CudaZFloatTruncatedPoissonLikelihood_GradientFactor
        else:
            raise NotImplementedError()

        I_target, valid_mask = ctx.saved_tensors
        gf = ctx.I_model

        gradient_factor(gf, I_target, valid_mask)

        # gf1 = fftshift(gf.cpu().numpy(),axes=[1,2])
        # p.var['gf1'] = gf1

        io.logger.debug('FarFieldPoissonLikelihood backward 2')
        # broadcast the gradient factor
        GS = np.array(grad_input.size())
        for dim in range(1, grad_input.ndimension() - 2):
            GS[dim] = 1
        GS = th.Size(GS)
        grad_input = grad_input * gf.view(GS).expand_as(grad_input)
        # grad_dpos = grad_input.clone()
        # x = grad_input.cpu().numpy()
        # p.var['gifft0'] = fftshift(x[:, 0, 0, :, :], axes=(1, 2))
        # x = grad_input.cpu().numpy()
        # p.var['go_%d' % TruncatedFarFieldPoissonLikelihood.i] = fftshift(x[:,0,0,:,:],axes=(1,2))
        # x = grad_input.cpu().numpy()
        # p.var['gi_%d' % TruncatedFarFieldPoissonLikelihood.i] = fftshift(x[:,0,0,:,:],axes=(1,2))

        # gi = grad_input.fftshift((3, 4))
        # x = grad_input.cpu().numpy()
        # p.var['grad_input.farfield_before_mask'] = fftshift(x[:, 0, 0, :, :], axes=(1, 2))
        if ctx.gradient_mask is not None:
            # x = ctx.gradient_mask.cpu().numpy()
            #        p.var['gifft2'] = fftshift(x[:, 0, 0, :, :]*valid_mask.cpu().numpy(), axes=(1, 2))
            # p.var['gradient_mask'] = fftshift(x)
            grad_input *= ctx.gradient_mask.expand_as(grad_input)

        # x = grad_input.cpu().numpy()
        # p.var['grad_input.farfield'] = fftshift(x[:, 0, 0, :, :], axes=(1, 2))

        grad_input.ifft2_()
        gi = grad_input.fftshift((3, 4))

        io.logger.debug('FarFieldPoissonLikelihood backward 3')

        grad_input = Variable(gi)
        return grad_input, None, None, None, None, None, None, None, None, None, None, None, None, None
