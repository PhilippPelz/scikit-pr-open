from skpr.core.models import CodedMeasurementNet
from . import *


class FarfieldCodedMeasurementNet(Module):
    def __init__(self, K, M, N, m_init=None, parallel_type='none'):
        super(FarfieldCodedMeasurementNet, self).__init__()
        self.coded_measurements = CodedMeasurementNet(K, M, N, m_init, parallel_type)

    def forward(self, input):
        io.logger.debug('FarfieldCodedMeasurements forward 1')
        exit_waves = self.coded_measurements(input)
        io.logger.debug('FarfieldCodedMeasurements forward 2')
        return exit_waves
