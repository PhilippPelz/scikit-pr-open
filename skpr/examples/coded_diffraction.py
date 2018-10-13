#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 10:11:21 2017

@author: philipp
"""

from skpr.core.engines import CodedMeasurementEngine
from skpr.core.models import FarfieldCodedMeasurementNet
from skpr.core.parameters import *
from skpr.inout.h5rw import h5read
from skpr.nn import modules as M
from skpr.util import *

p = get_default_parameters()

f = h5read('/home/pelzphil/projects/cd_sim.h5')

p.y = th.from_numpy(f['I'])

p.model = FarfieldCodedMeasurementNet

p.object.solution = th.from_numpy(f['ob'].astype(np.complex64))
p.object.update_start = 0
p.object.margins = 0

p.loss.function = M.TruncatedFarFieldPoissonLikelihood
p.loss.parameters.a_h = 10

gradient_mask_radius = 37
gradient_mask_falloff = 1

p.optimizer.object.type = th.optim.SGD
p.optimizer.object.parameters.lr = 1e-1 + 0j

p.logging.level = INFO
p.logging.log_reconstruction_parameters = False
p.logging.log_object_progress = False
p.logging.log_probe_progress = False
p.logging.log_error_progress = True
p.logging.print_summary = True
p.logging.print_report = True

p.epochs = 500

eng = CodedMeasurementEngine(p)
eng.fit()


