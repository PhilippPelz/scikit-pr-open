import numpy as np
import torch as th
from torch.optim.optimizer import Optimizer, required
import skpr.inout as io


class GradientDescent(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, max_iters=20, search_method='ncg',
                 update_objective_period=500, beta_choice='pr'):
        # maxIters = group['max_iters']
        # momentum = group['momentum']
        # dampening = group['dampening']
        # nesterov = group['nesterov']
        # search_method = group['search_method']
        # update_objective_period = group['update_objective_period']
        defaults = dict(lr=lr, max_iters=max_iters, search_method=search_method,
                        update_objective_period=update_objective_period, beta_choice=beta_choice)
        super(GradientDescent, self).__init__(params, defaults)
        self.last_ncg_reset_iter = 0
        self.current_solve_time = 0
        self.current_measurement_error = []
        self.current_residual = []
        self.current_recon_error = []
        self.solve_times = 0
        self.residuals = 0
        self.measurement_errors = 0
        self.recon_errors = 0
        self.search_dir0 = 0
        self.search_dir1 = 0
        self.gradf0 = 0
        self.gradf1 = 0
        self.maxDiff = 0
        self.last_objective_update_iter = 0
        self.tau1 = 0
        self.unscaled_search_dir = 0

    def __setstate__(self, state):
        super(GradientDescent, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def determine_initial_stepsize(self, p, closure):
        x_1 = th.randn(self.x0.size())
        x_2 = th.randn(self.x0.size())

        tmp = p.data.clone()

        p.data.copy_(x_1)
        loss = closure()
        gradf_1 = p.grad.data.clone()

        p.data.copy_(x_2)
        loss = closure()
        gradf_2 = p.grad.data.clone()

        p.data.copy_(tmp)
        L = th.norm(gradf_1 - gradf_2) / th.norm(x_2 - x_1)
        L = max(L, 1.0e-30)
        tau = 25.0 / L
        return tau

    def process_iteration(self):

        return False

    def determine_search_direction(self, beta_choice):
        if self.state.searchMethod == 'steepestdescent':
            search_dir = -self.gradf1
        if self.state.searchMethod == 'ncg':
            search_dir = -self.gradf1

            # Reset NCG progress after specified number of iterations have
            # passed
            if self.i - self.last_ncg_reset_iter == self.state.ncgResetPeriod:
                unscaled_search_dir = th.zeros(self.n, 1)
                self.last_ncg_reset_iter = self.i

            # Proceed only if reset has not just occurred
            if iter != self.last_ncg_reset_iter:
                if beta_choice.lower() == 'hs':
                    # Hestenes-Stiefel
                    beta = -self.gradf1.dot(self.Dg).real / self.unscaled_search_dir.dot(self.Dg).real
                elif beta_choice.lower() == 'fr':
                    # Fletcher-Reeves
                    beta = th.norm(self.gradf1) ** 2 / th.norm(self.gradf0) ** 2
                elif beta_choice.lower() == 'pr':
                    # Polak-Ribi�re
                    beta = self.gradf1.dot(self.Dg).real / th.norm(self.gradf0) ** 2
                elif beta_choice.lower() == 'dy':
                    # Dai-Yuan
                    beta = th.norm(self.gradf1) ^ 2 / self.unscaled_search_dir.dot(self.Dg).real
            search_dir = search_dir + beta * unscaled_search_dir
            self.unscaled_search_dir = search_dir

        if self.state.searchMethod == 'lbfgs':
            search_dir = -self.gradf1
            iters = np.min(iter - self.last_objective_update_iter, self.state.storedVectors)

            if iters > 0:
                alphas = th.zeros(iters, 1)

                # First loop
                for j in range(iters):
                    alphas[j] = self.rhoVals[j] * self.sVals[:, j].dot(search_dir).real
                    search_dir = search_dir - alphas(j) * self.yVals[:, j]

                # Scaling of search direction
                gamma = self.Dg.dot(self.Dx).real / self.Dg.dot(self.Dg).real
                search_dir = gamma * search_dir

                # Second loop
                for j in np.arange(iter, 1, -1):
                    beta = self.rhoVals[j] * self.yVals[:, j].dot(search_dir).real
                    search_dir = search_dir + (alphas[j] - beta) * self.sVals[:, j]

                search_dir = 1 / gamma * search_dir
                search_dir = th.norm(self.gradf1) / th.norm(search_dir) * search_dir

        # Change search direction to steepest descent direction if current
        # direction is invalid
        isNaN = search_dir != search_dir

        if th.any(isNaN):  # || any(isinf(searchDir)):
            search_dir = -self.gradf1

        # Scale current search direction match magnitude of gradient
        search_dir = th.norm(self.gradf1) / th.norm(search_dir) * search_dir
        return search_dir

    def updateStepsize(self):
        Ds = self.searchDir0 - self.searchDir1
        dotprod = th.dot(self.Dx.view(-1), Ds.view(-1)).real
        tauS = th.norm(self.Dx) ** 2 / dotprod  # First BB stepsize rule
        tauM = dotprod / th.norm(Ds) ** 2  # Alternate BB stepsize rule
        tauM = max(tauM, 0)
        if 2 * tauM > tauS:  # Use "Adaptive"  combination of tau_s and tau_m
            self.tau1 = tauM
        else:
            self.tau1 = tauS - tauM / 2  # Experiment with this param
        if self.tau1 <= 0:  # Make sure step is non-negative
            self.tau1 = self.tau0 * 1.5  # let tau grow, backtracking will kick in if stepsize is too big

    def step(self, closure=None):

        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        for group in self.param_groups:
            maxIters = group['max_iters']
            search_method = group['search_method']
            update_objective_period = group['update_objective_period']
            beta_choice = group['beta_choice']
            last_objective_update_iter = 0
            tolerance_penalty = 0
            update_objective_now = True
            max_diff = -np.inf

            if not group.recordTimes:
                self.solve_times = th.zeros(maxIters, 1)
            if not group.recordResiduals:
                self.residuals = th.zeros(maxIters, 1)
            if not group.recordMeasurementErrors:
                self.measurement_errors = th.zeros(maxIters, 1)
            if not group.recordReconErrors:
                self.recon_errors = th.zeros(maxIters, 1)
            for p in group['params']:
                loss = None
                x0 = p.data
                n = x0.numel()

                x1 = x0

                for i in range(maxIters):
                    if i - self.last_objective_update_iter == update_objective_period:
                        update_objective_now = True
                    if update_objective_now:
                        update_objective_now = False
                        self.last_objective_update_iter = i

                        p.data.copy_(x1)
                        loss = closure()
                        f1 = loss.data
                        loss.backward()
                        self.gradf1 = p.data.grad.clone()

                        if search_method == 'lbfgs':
                            # Perform LBFGS initialization
                            yVals = th.zeros(n, group.storedVectors)
                            sVals = th.zeros(n, group.storedVectors)
                            rhoVals = th.zeros(1, group.storedVectors)
                        elif search_method == 'ncg':
                            # Perform NCG initialization
                            self.last_ncg_reset_iter = i
                            self.unscaled_search_dir = th.zeros(n, 1)

                        self.search_dir1 = self.determine_search_direction(beta_choice)
                        # Reinitialize stepsize to supplement new objective function
                        self.tau1 = self.determine_initial_stepsize()
                    else:
                        p.data.copy_(x1)
                        loss = closure()
                        gradf1 = p.data.grad.clone()
                        Dg = self.gradf1 - self.gradf0

                        if search_method == 'lbfgs':
                            # Update  LBFGS stored vectors
                            sVals = th.cat([self.Dx, sVals[:, :group.storedVectors - 1]], 0)
                            yVals = th.cat([Dg, yVals[:, :group.storedVectors - 1]], 0)
                            rhoVals = th.cat(
                                [1 / Dg.view(-1).dot(self.Dx.view(-1)).real, rhoVals[:, :group.storedVectors - 1]], 0)

                        self.search_dir1 = self.determine_search_direction()
                        self.updateStepsize()

                    x0 = x1
                    f0 = f1
                    self.gradf0 = gradf1
                    self.tau0 = self.tau1
                    self.search_dir0 = self.search_dir1

                    x1 = x0 + tau0 * self.search_dir0
                    self.Dx = x1 - x0

                    p.data.copy_(x1)
                    loss = closure()
                    f1 = loss

                    # We now determine an appropriate stepsize for our algorithm using
                    # Armijo-Goldstein condition

                    backtrack_count = 0
                    # Cap maximum number of backtracks

                    while backtrack_count <= 3:
                        tmp = f0 + 0.1 * tau0 * self.search_dir0.view(-1).dot(self.gradf0.view(-1)).real

                        # Break if f1 < tmp or f1 is sufficiently         close         to         tmp(determined error)
                        # Avoids         division         by         zero
                        if f1 <= tmp:
                            break

                        backtrack_count = backtrack_count + 1
                        # Stepsize  reduced  by  factor  of  5
                        tau0 = tau0 * 0.2
                        x1 = x0 + tau0 * self.search_dir0
                        self.Dx = x1 - x0
                        p.data.copy_(x1)
                        f1 = closure()

                    # Handle     processing     of     current     iteration     estimate
                    stop_now = self.process_iteration()
                    if stop_now:
                        break

        def display_verbose_output():
            print('Iter = %d', iter)
            print(' | IterationTime = %.3f' % self.current_solve_time)
            print(' | Resid = %.3e' % self.current_residual)
            print(' | Stepsize = %.3e' % self.tau0)
            if self.current_measurement_error is not None:
                print(' | M error = %.3e' % self.current_measurement_error)
            if self.current_recon_error is not None:
                print(' | R error = %.3e' % self.current_recon_error)
            print
