import datetime
import os

import torch.optim.lr_scheduler as lrs

import skpr.nn.modules as M
from skpr.core.ptycho.models import *
from skpr.util import Param

CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0

def on(self):
    for k in self:
        if isinstance(self[k], bool):
            self[k] = True

def off(self):
    for k in self:
        if isinstance(self[k], bool):
            self[k] = False


def get_default_parameters():
    p = Param()

    p.log_folder = os.getcwd()
    p.log_interval = 1
    p.NP = 1
    p.NO = 1
    p.epochs = 50
    p.save_file_name = 'ptycho.h5'

    p.model = None

    p.loss = Param()
    p.loss.function = M.PoissonLikelihood
    p.loss.function_parameters = Param()
    p.loss.function_lr_schedule = lambda epoch: 1.0
    p.loss.object_prior_lr_schedule = lambda epoch: 1.0
    p.loss.object_prior = None
    p.loss.object_prior_parameters = Param()
    p.loss.object_prior_start_epoch = 1
    p.loss.object_prior_steps_per_epoch = 1
    p.loss.probe_centering_prior = None

    p.cuda = True
    p.parallel_type = 'none'  # none  local  distributed
    p.world_size = 1
    p.dist_backend = 'gloo'
    p.dist_url = 'tcp://224.66.41.62:23456'

    p.calculate_dose_from_probe = False

    p.save = Param()
    p.save.interval = 5
    p.save.path = os.getcwd()
    p.save.save_raw_data = True
    p.save.when_finished = True
    # p.run_label = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    p.run_label = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    p.descriptive_string = ''

    p.regularizer = Param()
    p.regularizer.type = None
    p.regularizer.parameters = Param()

    p.optimizer = Param()
    p.optimizer.object = Param()
    p.optimizer.object.type = th.optim.SGD
    p.optimizer.object.parameters = Param()
    p.optimizer.object.parameters.lr = 1
    p.optimizer.object.lr_scheduler = Param()
    p.optimizer.object.lr_scheduler.type = lrs.LambdaLR
    p.optimizer.object.lr_scheduler.parameters = Param()
    p.optimizer.object.lr_scheduler.parameters.lr_lambda = lambda epoch: 1.0

    p.optimizer.object_prior = Param()
    p.optimizer.object_prior.parameters = Param()
    p.optimizer.object_prior.type = th.optim.SGD
    p.optimizer.object_prior.parameters.lr = 1

    p.optimizer.object_prior.lr_scheduler = Param()
    p.optimizer.object_prior.lr_scheduler.type = lrs.LambdaLR
    p.optimizer.object_prior.lr_scheduler.parameters = Param()
    p.optimizer.object_prior.lr_scheduler.parameters.lr_lambda = lambda epoch: 1.0

    p.optimizer.n_batches = 4
    p.optimizer.batch_selection = 'random'  # 'cluster'

    p.optimizer.optimize_probe_and_object_jointly = lambda epoch: False
    p.optimizer.alpha = lambda epoch: 0.05 if epoch < 30 else 1
    p.optimizer.beta = 0.9

    p.valid_mask = None
    p.gradient_mask = None

    p.object = Param()
    p.object.initial = None
    p.object.solution = None
    p.object.update_start = 0
    p.object.margins = 5

    p.logging = Param()
    p.logging.log_file_path = os.getcwd() + '/log/%s.log' % p.run_label
    p.logging.log_reconstruction_parameters = False
    p.logging.log_object_progress = True
    p.logging.log_probe_progress = True
    p.logging.log_error_progress = True
    p.logging.on = on
    p.logging.off = off
    p.logging.level = INFO
    p.logging.print_summary = True
    p.logging.print_report = True

    p.experiment = Param()
    p.experiment.z = 0.7
    p.experiment.E = 300e3
    p.experiment.det_pix = 70e-6
    p.experiment.N_det_pix = 128

    return p


def get_ptycho_default_parameters():
    p = get_default_parameters()

    p.plot_clusters = False

    p.ptycho = Param()
    p.ptycho.pos_solution = None

    p.optimizer.probe = Param()
    p.optimizer.probe.type = th.optim.SGD
    p.optimizer.probe.parameters = Param()
    p.optimizer.probe.parameters.lr = 1

    p.optimizer.position = Param()
    p.optimizer.position.type = th.optim.SGD
    p.optimizer.position.parameters = Param()
    p.optimizer.position.parameters.lr = 1e-5

    p.optimizer.overlapping_probes = False

    p.valid_mask = None

    p.model = PtychoNet

    p.probe = Param()
    p.probe.initial = None
    p.probe.update_start = 4
    p.probe.support_radius = None
    p.probe.support_radius_fourier = None
    p.probe.keep_intensity = True
    p.probe.solution = None
    p.probe.scale_intensity_to_data = False
    p.probe.gradient_mask = None
    p.probe.start_apply_gradient_mask = 1e5
    p.probe.amplitude_constraint = None
    p.probe.centering_active = lambda epoch: True if epoch < 5 else False
    p.probe.subpixel_precision_active = lambda epoch: True if epoch > 0 else False
    p.probe.subpixel_optimization_start = 1e3
    p.probe.subpixel_optimization_iterations = 50
    p.probe.subpixel_optimization_trials = 4
    p.probe.subpixel_optimization_max_displacement = 2
    p.probe.subpixel_optimization_every = 2
    p.probe.subpixel_optimization_loss = M.SingleFarFieldPoissonLikelihood
    p.probe.positions_positive = False
    p.probe.fix_dead_probe_pixels = False
    p.probe.radius = -1

    p.loss.probe_centering_prior_parameters = Param()
    p.loss.probe_centering_prior_parameters.radius_fraction = 0.26
    p.loss.probe_centering_prior_parameters.falloff_pixels = 12

    return p


