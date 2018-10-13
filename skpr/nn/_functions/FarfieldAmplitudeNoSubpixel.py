# -*- coding: utf-8 -*-
import skpr.inout  as io
import numpy as np
import torch as th
from torch.autograd import Function, Variable


class FarFieldAmplitudeNoSubpixel(Function):
    @staticmethod
    def forward(ctx, wave, gradient_mask):
        """
        Parameters
        ----------
        wave : dimension: K, N_p, N_o, M1, M2

        Returns
        -------
        I_model : dimension: K, M1, M2 float tensor
            diffraction intensities
        """
        io.logger.debug('FarFieldAmplitudeNoSubpixel forward 1')

        # I = np.abs(wave.cpu().numpy()[:, 0, 0, :, :])
        # io.plotmosaic(I, 'abs exit wave')

        wave_shifted = wave.ifftshift((3, 4))
        wave_farfield = wave_shifted.fft2_()

        ctx.wave_farfield = wave_farfield
        ctx.gradient_mask = gradient_mask

        I_model = th.cuda.FloatTensor(wave.size())
        wave_farfield.expect(out=I_model)

        # I = I_model.cpu().numpy()[:, 0, 0, :, :]
        # io.plotmosaic(I, 'I_model')

        # sum up all dimensions but the one with indices 0 (batch dimension) and size-1, size-2 (image dimensions)
        for dim in range(1, I_model.ndimension() - 2):
            I_model = th.sum(I_model, dim, keepdim=True)
        io.logger.debug('FarFieldAmplitudeNoSubpixel forward 2')
        return I_model.squeeze().sqrt_()

    @staticmethod
    def backward(ctx, grad_output):
        """
        backward.
        Parameters
        ----------
        grad_output : (K, Mx, My) float tensor

        Returns
        -------
        grad_input : (K, NP, NO, Mx, My) complex tensor
        grad_dpos : (K, NP, NO, Mx, My) complex tensor
        """
        io.logger.debug('FarFieldAmplitudeNoSubpixel backward 1')

        grad_input = ctx.wave_farfield.clone()

        GS = np.array(grad_input.size())
        for dim in range(1, grad_input.ndimension() - 2):
            GS[dim] = 1
        GS = th.Size(GS)
        grad_input *= grad_output.data.view(GS).expand_as(grad_input)

        if ctx.gradient_mask is not None:
            # x = ctx.gradient_mask.cpu().numpy()
            #        p.var['gifft2'] = fftshift(x[:, 0, 0, :, :]*valid_mask.cpu().numpy(), axes=(1, 2))
            # p.var['gradient_mask'] = fftshift(x)
            grad_input *= ctx.gradient_mask.expand_as(grad_input)

        grad_input.ifft2_()
        grad_input = grad_input.fftshift((3, 4))

        Vgrad_input = Variable(grad_input)
        io.logger.debug('FarFieldAmplitudeNoSubpixel backward 1')
        return Vgrad_input, None
