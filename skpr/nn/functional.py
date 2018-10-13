# -*- coding: utf-8 -*-
from ._functions.BM3DPrior import BM3DPrior
from ._functions.Broadcast import Broadcast
from ._functions.CMul import CMul
from ._functions.CMulNoSubpixel import CMulNoSubpixel
from ._functions.CenteringPrior import CenteringPrior
from ._functions.CropAtPositions import CropAtPositions
from ._functions.FFT import FFT
from ._functions.FFTShift import FFTShift
from ._functions.FarFieldPoissonLikelihood import FarFieldPoissonLikelihood
from ._functions.FarfieldIntensity import FarFieldIntensity
from ._functions.FarfieldIntensityNoSubpixel import FarFieldIntensityNoSubpixel
# from ._functions.FarfieldAmplitude import FarfieldAmplitude
# from ._functions.FarfieldAmplitudeNoSubpixel import FarfieldAmplitudeNoSubpixel
from ._functions.IFFT import IFFT
from ._functions.PoissonNoise import PoissonNoise
from ._functions.SingleFarFieldPoissonLikelihood import SingleFarFieldPoissonLikelihood
from ._functions.SubpixelShift import SubpixelShift
from ._functions.SubpixelShiftNoShiftGrad import SubpixelShiftNoShiftGrad
from ._functions.TruncatedFarFieldPoissonLikelihood import TruncatedFarFieldPoissonLikelihood
from ._functions.PoissonLikelihood import PoissonLikelihood
from ._functions.TruncatedPoissonLikelihood import TruncatedPoissonLikelihood
from ._functions.SinglePoissonLikelihood import SinglePoissonLikelihood
from ._functions.AmplitudeLoss import AmplitudeLoss

broadcast = Broadcast.apply
cmul = CMul.apply
cmul_no_subpixel_gradient = CMulNoSubpixel.apply
crop_at_positions = CropAtPositions.apply
truncated_farfield_poisson_likelihood = TruncatedFarFieldPoissonLikelihood.apply
farfield_poisson_likelihood = FarFieldPoissonLikelihood.apply
single_farfield_poisson_likelihood = SingleFarFieldPoissonLikelihood.apply
farfield_intensity = FarFieldIntensity.apply
farfield_intensity_no_subpixel_gradient = FarFieldIntensityNoSubpixel.apply
# farfield_amplitude = FarfieldAmplitude.apply
# farfield_amplitude_no_subpixel_gradient = FarfieldAmplitudeNoSubpixel.apply
poisson_noise = PoissonNoise.apply
poisson_likelihood = PoissonLikelihood.apply
truncated_poisson_likelihood = TruncatedPoissonLikelihood.apply
single_poisson_likelihood = SinglePoissonLikelihood.apply
bm3d_prior = BM3DPrior.apply
centering_prior = CenteringPrior.apply
subpixel_shift = SubpixelShift.apply
subpixel_shift_no_shift_gradients = SubpixelShiftNoShiftGrad.apply
fft2 = FFT.apply
ifft2 = IFFT.apply
fftshift = FFTShift.apply
amplitude_loss = AmplitudeLoss.apply