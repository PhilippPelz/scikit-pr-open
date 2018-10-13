# -*- coding: utf-8 -*-

import torch as th
from torch.autograd import Function


# void CudaZFloatTruncatedPoissonLikelihood_updateOutput(THCudaTensor *input, THCudaTensor *target, THCudaTensor *mask, THCudaTensor *output);
# void CudaZFloatTruncatedPoissonLikelihood_GradientFactor(THCudaTensor *input, THCudaTensor *target, THCudaTensor *mask);
# void CudaZDoubleTruncatedPoissonLikelihood_updateOutput(THCudaDoubleTensor *input, THCudaDoubleTensor *target, THCudaDoubleTensor *mask, THCudaDoubleTensor *output);
# void CudaZDoubleTruncatedPoissonLikelihood_GradientFactor(THCudaDoubleTensor *input, THCudaDoubleTensor *target, THCudaDoubleTensor *mask);

class PoissonNoise(Function):
    @staticmethod
    def forward(ctx, intensity):
        """
        input       dimension: K, N_o, N_p, M1, M2
        I_target    dimension: K, M1, M2
        mask        dimension: K, M1, M2

                Should be overriden by all subclasses.
        """
        # print 'FarFieldmentedError()

        I = intensity.cpu().numpy()
        # print I.shape
#        print I.max()
        #         plot(np.log10(I[0]),'I')
        # I_noisy = poisson(I)
        I_noisy = I
        # plot(np.log10(I_noisy[0]),'I_noisy')
        return th.from_numpy(I_noisy).cuda()

    @staticmethod
    def backward(ctx, grad_output):
        # print 'FarFieldPoissonLikelihood.backward'

        raise NotImplementedError()

        return None, None, None
