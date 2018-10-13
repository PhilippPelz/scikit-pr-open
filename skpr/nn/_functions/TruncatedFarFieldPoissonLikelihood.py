# -*- coding: utf-8 -*-

import numpy as np
import torch as th
from torch.autograd import Function, Variable

# void CudaZFloatTruncatedPoissonLikelihood_updateOutput(THCudaTensor *input, THCudaTensor *target, THCudaTensor *valid_mask, THCudaTensor *output);
# void CudaZFloatTruncatedPoissonLikelihood_GradientFactor(THCudaTensor *input, THCudaTensor *target, THCudaTensor *valid_mask);
# void CudaZDoubleTruncatedPoissonLikelihood_updateOutput(THCudaDoubleTensor *input, THCudaDoubleTensor *target, THCudaDoubleTensor *valid_mask, THCudaDoubleTensor *output);
# void CudaZDoubleTruncatedPoissonLikelihood_GradientFactor(THCudaDoubleTensor *input, THCudaDoubleTensor *target, THCudaDoubleTensor *valid_mask);
import skpr.inout  as io
from skpr._ext import skpr_thnn


class TruncatedFarFieldPoissonLikelihood(Function):
    i = 0
    
    @staticmethod
    def forward(ctx, input, I_target, beam_amplitude, valid_mask, gradient_mask, a_h, M, NP, NO):
        """
        input               dimension: K, N_o, N_p, M1, M2
        I_target            dimension: K, M1, M2
        valid_mask          dimension: K, M1, M2

                Should be overriden by all subclasses.
        """
        # print 'FarFieldPoissonLikelihood.forward'
        if 'ZFloat' in type(input).__name__:
            likelihood = skpr_thnn.CudaZFloatTruncatedPoissonLikelihood_updateOutput
        elif 'ZDouble' in type(input).__name__:
            likelihood = skpr_thnn.CudaZDoubleTruncatedPoissonLikelihood_updateOutput
        else:
            raise NotImplementedError()
#        print input.norm()
        PsiF_model = input.fft2()
#        print PsiF_model.norm()
        ctx.PsiF_model = PsiF_model
        ctx.a_h = a_h
        ctx.M = M
        ctx.NP = NP
        ctx.NO = NO
        ctx.probe_amplitude = beam_amplitude

        # PsiF_model.mul_(beam_amplitude[0])
        I_model = th.cuda.FloatTensor(input.size())

        #        print 'I_model',dir(I_model)
        #        print dir(PsiF_model)
        PsiF_model.expect(out=I_model)

        # print('I_model.sum() ', I_model.sum())
        # print('I_target.sum()', I_target.sum())

        # print(I_model.size())
        # print(I_target.size())
        # io.logger.info('i = %d' % i)
        #        if i == 4500:
        #            I = fftshift(I_model.squeeze().cpu().numpy(), axes=[0, 1])
        #            It = fftshift(I_target.cpu().numpy(), axes=[0, 1])
#        plot(np.log10(I[0,0,0]))
#        plot(np.log10(It[0]))
        #         print np.sum(I[633,0,0]), np.sum(It[633])
        #            zplot([np.log10(I), np.log10(It)], 'model  - target', cmap=['hot', 'hot'])
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

        # TODO: make this cpu
        out = th.cuda.FloatTensor(1)
        # print I_model.size()
        # print I_target.size()
        # print valid_mask.size()
        likelihood(I_model, I_target, valid_mask, out)
        io.logger.debug('TruncatedFarFieldPoissonLikelihood out = %3.2g' % out[0])
        return out.cpu()

    @staticmethod
    def backward(ctx, grad_output):
        TruncatedFarFieldPoissonLikelihood.i = TruncatedFarFieldPoissonLikelihood.i+1
        io.logger.debug('TruncatedFarFieldPoissonLikelihood backward 1')
        PsiF_model = ctx.PsiF_model
        a_h = ctx.a_h
        grad_input = PsiF_model.clone()
        if 'ZFloat' in type(PsiF_model).__name__:
            gradient_factor = skpr_thnn.CudaZFloatTruncatedPoissonLikelihood_GradientFactor
        elif 'ZDouble' in type(PsiF_model).__name__:
            gradient_factor = skpr_thnn.CudaZFloatTruncatedPoissonLikelihood_GradientFactor
        else:
            raise NotImplementedError(
                'TruncatedPoissonLikelihood_GradientFactor is not implemented for dtype ' + type(PsiF_model).__name_)
            
        I_target, valid_mask = ctx.saved_tensors
        gf = ctx.I_model 

        K_t = (gf - I_target).norm(1)/I_target.numel()
        lhs = (gf - I_target).abs()

        io.logger.debug('TruncatedFarFieldPoissonLikelihood K_t: %3.2g' % K_t)
        io.logger.debug('TruncatedFarFieldPoissonLikelihood a_h: %3.2g' % a_h)

        truncation_mask = (lhs <= K_t * a_h).float()
        # print 'truncation_mask size', truncation_mask.size()
        # p.var['truncation_mask'] = fftshift(truncation_mask.cpu().float().numpy(), axes=[1,2])

        # gf1 = fftshift(gf.cpu().numpy(),axes=[1,2])
        # p.var['gf1_%d' % TruncatedFarFieldPoissonLikelihood.i] = gf1
        
        gradient_factor(gf, I_target, valid_mask)
        # th.cuda.synchronize()
        # gf1 = fftshift(gf.cpu().numpy(),axes=[1,2])
        # p.var['gf1_%d' % TruncatedFarFieldPoissonLikelihood.i] = gf1

        io.logger.debug('TruncatedFarFieldPoissonLikelihood backward 2')
        # broadcast the gradient factor
        GS = np.array(grad_input.size())
        for dim in range(1, grad_input.ndimension() - 2):
            GS[dim] = 1
        GS = th.Size(GS)
        io.logger.debug('TruncatedFarFieldPoissonLikelihood backward 2.1')

        gf = gf.view(GS).expand_as(grad_input)
        truncation_mask = truncation_mask.view(GS).expand_as(grad_input)

        grad_input = grad_input * gf * truncation_mask
        
        # grad_input.mul_(gradient_mask.expand_as(grad_input))
        #grad_input.mul_(truncation_mask.expand_as(grad_input))
        # x = grad_input.cpu().numpy()
        # p.var['go_%d' % TruncatedFarFieldPoissonLikelihood.i] = fftshift(x[:,0,0,:,:],axes=(1,2))
        # x = grad_input.cpu().numpy()
        # p.var['gi_%d' % TruncatedFarFieldPoissonLikelihood.i] = fftshift(x[:,0,0,:,:],axes=(1,2))
        io.logger.debug('TruncatedFarFieldPoissonLikelihood backward 2.2')
        grad_input.ifft2_()
        # grad_input.div_(probe_amplitude[0])
        io.logger.debug('TruncatedFarFieldPoissonLikelihood backward 3')

        grad_input = Variable(grad_input)
        # x = grad_input.data.cpu().numpy()
        # p.var['gifft_%d' % TruncatedFarFieldPoissonLikelihood.i] = x[:,0,0,:,:]
        return grad_input, None, None, None, None, None, None, None, None, None, None, None, None
