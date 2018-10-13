import datetime
import os
import socket
from collections import OrderedDict

import torch.utils.hooks as hooks
from tensorboardX import SummaryWriter
from torch.autograd import Variable

import skpr.inout as io
import skpr.nn as p
from skpr.util import *


class BaseEngine(object):
    def __init__(self, params):
        self.epoch = 0
        self.N = [0, 0]
        self.epoch = np.zeros(1)

        self.copy_stream = th.cuda.Stream()

        self._parameters = params
        self._post_iteration_hooks = OrderedDict()
        self._pre_iteration_hooks = OrderedDict()
        self.writer = SummaryWriter(
            log_dir=os.path.join('runs', datetime.datetime.now().strftime(
                '%d_%H-%M-%S') + '_' + socket.gethostname()) + self.descriptive_string)
        self.log_params = Param()
        self.log_params.loss = [-1]
        self.log_params.object_prior_loss = [-1]
        self.log_params.probe_centering_prior_loss = [-1]
        self.log_params.object_gradient_norm = [-1]

        if self.object.solution is not None:
            self.log_params.object_error = [-1]

        self.set_up_logging()

        self.K = self.pos.size()[0]
        self.M = np.array([self.y.size()[1], self.y.size()[2]])

        self.has_object_solution = self.object.solution is not None

        # order is important
        self.determine_image_size()
        self.allocate_variables()
        self.maybe_log_parameters()
        self.maybe_initialize_solutions()
        self.initialize_modules()

        if not self.loss.function.is_intensity_based:
            # loss is amplitude based, convert intensities to amplitudes
            self.y.sqrt_()

        # print 'parameters: '
        # for n,p in self.model.named_parameters():
        #     print n

        self.initialize_optimizers()

    def initialize_optimizers(self):
        pass

    def maybe_initialize_solutions(self):
        pass

    def determine_image_size(self):
        pass

    def initialize_modules(self):
        pass

    def allocate_variables(self):
        self.measurements = Variable(data=self.y.cuda(), requires_grad=False)

    def maybe_calculate_errors(self, epoch):
        pass

    def get_object(self):
        pass

    def maybe_log(self, epoch):
        pass

    def set_up_logging(self):
        io.init_logging(self._parameters)

    def maybe_log_parameters(self):
        if self.logging.log_reconstruction_parameters:
            io.logger.info('------------------------------------------------------------------------------')
            io.logger.info('Reconstruction parameters:')
            # io.logger.info(str(self._parameters))
            io.logger.info('------------------------------------------------------------------------------')

    def maybe_print_summary_header(self):
        label_format = OrderedDict()
        label_format['it'] = '%-4s'
        label_format['L'] = '%-15s'
        label_format['L_op'] = '%-15s'
        label_format['L_pc'] = '%-15s'
        label_format['||dL/dO||'] = '%-15s'

        self.add_summary_header_entries(label_format)

        formats = ''
        labels = ()
        for k, v in label_format.iteritems():
            formats += v
            labels += (k,)

        if self.logging.print_summary:
            io.logger.info(formats % labels)

    def maybe_print_summary(self, epoch):
        label_format = [(epoch, '%-4d'), \
                        (self.log_params.loss[-1], '%-15g'), \
                        (self.log_params.object_prior_loss[-1], '%-15g'), \
                        (self.log_params.probe_centering_prior_loss[-1], '%-15g'), \
                        (self.log_params.object_gradient_norm[-1], '%-15g')]

        self.add_summary_entries(label_format)

        formats = ''
        labels = ()
        for k, v in label_format:
            formats += v
            labels += (k,)
        if self.logging.print_summary:
            io.logger.info(formats % labels)

    def maybe_print_report(self):
        pass

    def loop_step(self):
        pass

    def before_epoch_loop(self):
        pass

    def after_epoch_loop(self):
        pass

    def define_clusters(self):
        pass

    def fit(self):
        self.before_epoch_loop()
        self.maybe_define_clusters()
        for epoch in range(self.epochs):
            self.epoch[0] = epoch
            p.var['epoch'] = epoch

            for hook in self._pre_iteration_hooks.values():
                hook(self)

            self.loop_step(epoch)

            self.maybe_calculate_errors(epoch)
            self.maybe_log(epoch)
            self.maybe_print_summary(epoch)

            for hook in self._post_iteration_hooks.values():
                hook(self)
        self.maybe_save_result()
        self.after_epoch_loop()

    def maybe_save_result(self):
        pass

    def model_forward(self):
        exit_waves = self.model(self.P, self.pos, self.dpos)
        loss = self.L(exit_waves, self.measurements, self.valid_measurements)
        self.log_params.loss.append(loss.data[0])
        loss.backward()
        # prior_loss.backward()
        return loss.data[0]  # + prior_loss.data[0]

    def object_model_forward(self):
        exit_waves = self.model(self.P, self.pos, self.dpos)
        loss = self.L(exit_waves, self.measurements, self.valid_measurements)
        self.log_params.loss.append(loss.data[0])
        loss.backward()

        self.set_margins_zero(self.get_object().grad.data)

        return loss.data[0], self.get_object().grad.data

    def object_prior_forward(self, epoch):
        if self.object_prior is not None:
            prior_loss = self.object_prior(self.get_object(), epoch)
            self.log_params.object_prior_loss.append(prior_loss.data[0])
            prior_loss.backward()

            return prior_loss.data[0], self.get_object().grad.data
        else:
            return 0, 0

    def model_output(self):
        return None

    def register_pre_iteration_hook(self, hook):
        """Registers a hook.

        The hook will be called every time the gradients with respect to module
        inputs are computed. The hook should have the following signature::

            hook(engine) -> None

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.
        """
        handle = hooks.RemovableHandle(self._hooks)
        self._pre_iteration_hooks[handle.id] = hook
        return handle

    def register_post_iteration_hook(self, hook):
        """Registers a hook.

        The hook will be called every time the gradients with respect to module
        inputs are computed. The hook should have the following signature::

            hook(engine) -> None

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.
        """
        handle = hooks.RemovableHandle(self._hooks)
        self._post_iteration_hooks[handle.id] = hook
        return handle

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
            if 'ptycho' in _parameters:
                ptycho = self.__dict__['_parameters']['ptycho']
                if name in ptycho:
                    return ptycho[name]

        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))
