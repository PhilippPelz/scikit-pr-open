from skpr.core.models import CodedMeasurementNet
from . import *


class SimFarfieldCodedMeasurementNet(Module):
    def __init__(self, K, M, N, O_init=None, parallel_type='none'):
        super(SimFarfieldCodedMeasurementNet, self).__init__()
        self.coded_measurements = CodedMeasurementNet(K, M, N, O_init, parallel_type)

    def forward(self, input):
        io.logger.debug('FarfieldCodedMeasurements forward 1')
        m = self.coded_measurements(input)
        intensity = F.farfield_intensity(m, 1, False)
        #        I = intensity.cpu().data.numpy()
        # It = I_target.cpu().numpy()
        # print(I.dtype)
        # print(It.shape)
        # zplot([np.log10(I[0,0,0]),np.log10(It[0])], cmap=['hot','hot'])
        #        plot(I[1])
        #        plot(I[2])
        #        noisy_intensity = F.poisson_noise(intensity)
        noisy_intensity = intensity
        io.logger.debug('FarfieldCodedMeasurements forward 2')

        return noisy_intensity
