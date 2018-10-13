# -*- coding: utf-8 -*-

import numpy as np
import torch as th
from torch.autograd import Function, Variable
import skpr.inout  as io


class FarFieldIntensity(Function):
    @staticmethod
    def forward(ctx, wave, dpos, gradient_mask):
        """
        Parameters
        ----------
        wave : dimension: K, N_p, N_o, M1, M2
        dpos : dimension: (2, K, NP, NO, Mx, My) complex tensor

        Returns
        -------
        I_model : dimension: K, M1, M2 float tensor
            diffraction intensities
        """
        io.logger.debug('FarFieldIntensity forward 1')
        # I = np.abs(wave.cpu().numpy()[:, 0, 0, :, :])
        # io.plotmosaic(I, 'abs exit wave')
        wave_shifted = wave.ifftshift((3, 4))
        wave_farfield = wave_shifted.fft2_()
        ctx.wave_farfield = wave_farfield
        ctx.gradient_mask = gradient_mask

        I_model = th.cuda.FloatTensor(wave.size())
        wave_farfield.expect(out=I_model)

        # I = I_model.cpu().numpy()[:,0,0,:,:]
        # io.plotmosaic(I,'I_model')

        # sum up all dimensions but the one with indices 0 (batch dimension) and size-1, size-2 (image dimensions)
        for dim in range(1, I_model.ndimension() - 2):
            I_model = th.sum(I_model, dim, keepdim=True)
        io.logger.debug('FarFieldIntensity forward 2')
        return I_model.squeeze()

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
        grad_dpos : (2, K, NP, NO, Mx, My) complex tensor
        """
        io.logger.debug('FarFieldIntensity backward 1')

        grad_input = ctx.wave_farfield.clone()
        grad_dpos = th.cuda.ZFloatTensor(2, *ctx.wave_farfield.size())
        wave_conj = ctx.wave_farfield.conj()
        grad_dpos[0].copy_(wave_conj)
        grad_dpos[1].copy_(wave_conj)  # repeat has a bug for complex tensor, thus this copying

        gos = grad_output.size()
        grad_input *= grad_output.data.view(gos[0], 1, 1, gos[1], gos[2]).expand_as(grad_input)
        grad_dpos *= grad_output.data.view(1, gos[0], 1, 1, gos[1], gos[2]).expand_as(grad_dpos)
        io.logger.debug('FarFieldIntensity backward 2')

        if ctx.gradient_mask is not None:
            grad_input *= ctx.gradient_mask.expand_as(grad_input)

        grad_input.ifft2_()
        gi = grad_input.fftshift((3, 4))

        Vgrad_input = Variable(gi)
        Vgrad_dpos = Variable(grad_dpos)
        io.logger.debug('FarFieldIntensity backward 3')
        return Vgrad_input, Vgrad_dpos, None

        # return Vgrad_input, None, None
