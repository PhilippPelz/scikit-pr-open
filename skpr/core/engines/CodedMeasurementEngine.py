from torch.autograd import Variable

from skpr.core.engines import BaseEngine
from skpr.util import *


class CodedMeasurementEngine(BaseEngine):
    def __init__(self, params):
        super(CodedMeasurementEngine, self).__init__(params)
        self.initialize_object()

    def initialize_object(self):
        self.O.data.copy_(self.object.solution)

    def maybe_initialize_solutions(self):
        if self.object.solution is not None:
            # print (1, self.NO, self.N[0] + 2 * self.object.margins, self.N[1] + 2 * self.object.margins)
            s = th.Size((self.N[0], self.N[1]))
            O_copy = self.object.solution.clone()
            self._parameters.object.solution = self._parameters.object.solution.new().resize_(*s)
            self._parameters.object.solution.fill_(0 + 1j * 0)
            self._parameters.object.solution[:O_copy.size()[0], :O_copy.size()[1]].copy_(O_copy)

    def initialize_optimizers(self):
        # print self.optimizer.object.parameters.lr
        self.object_optimizer = self.optimizer.object.type([self.O], \
                                                           **self.optimizer.object.parameters)

    def initialize_modules(self):
        self.model = self.model(self.K, self.M, self.N, self.object.initial, self.parallel_type)
        # self.object_prior = M.rPIEPrior(self.probe.prior_lambda)
        self.loss.parameters.M = self.M
        self.loss.parameters.NO = self.NO
        self.loss.parameters.NP = self.NP
        self.loss.parameters.beam_amplitude = 1

        # print 'self.loss.parameter', self.loss.parameters
        self.L = self.loss.function(**self.loss.parameters)

        distributed = self.world_size > 1

        if distributed:
            dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url, world_size=self.world_size)

    def allocate_variables(self):
        self.measurements = Variable(data=self.y.cuda(), requires_grad=False)
        p = th.ZFloatTensor(self.N[0], self.N[1])
        p.pin_memory()
        self.O = Variable(data=p.cuda(), requires_grad=True)
        # print 'P is leaf: ', self.P.is_leaf

    def before_epoch_loop(self):
        self.maybe_print_report()
        self.maybe_print_summary_header()
        self.object_optimizer.zero_grad()

    def loop_step(self, epoch):
        if self.object.update_start <= epoch:
            # self.object_model_forward()
            # absmax = self.get_object().grad.abs().max()
            # self.get_object().grad.div_(absmax)
            # dO = self.get_object().grad.data.cpu().numpy()
            # zplot((np.abs(dO[0, 0]), np.angle(dO[0, 0])), 'dO')
            # dO = self.P.grad.data.cpu().numpy()
            # zplot((np.abs(dO[0,0]),np.angle(dO[0,0])), 'dP')
            print 'before optim step'
            self.object_optimizer.step(self.object_model_forward)

    def determine_image_size(self):
        s = self.y.size()
        self.N = [s[1], s[2]]

    def model_output(self):
        out = self.model(self.O)
        return out
