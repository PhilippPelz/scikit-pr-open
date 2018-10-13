from collections import deque

from torch.optim.optimizer import Optimizer, required
import skpr.inout as io

class HistorySGD(Optimizer):
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

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, T=5, steps_per_iteration=1, start_iteration=1000, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, T=T, history=deque(),
                        start_iteration=start_iteration)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(HistorySGD, self).__init__(params, defaults)
        self.step_in_iteration = 0
        self.steps_per_iteration = steps_per_iteration

    def __setstate__(self, state):
        super(HistorySGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()


        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            history = group['history']
            start_iteration = group['start_iteration']
            T = group['T']

            for p in group['params']:
                param_state = self.state[p]

                if p.grad is None:
                    continue
                d_p = p.grad.data
                p_now = p.data
                # if p_now.size()[2] > 1000:
                #     c = p_now[0, 0].cpu().numpy()
                #     io.plotAbsAngle(c, 'p_now')
                history.append(p_now)

                if momentum != 0 and len(history) >= T:
                    v_history = param_state['v_history']
                    v_hist = v_history.popleft()
                    p_hist = history.popleft()
                    # if p_hist.size()[2] > 1000:
                    #     c = p_hist[0, 0].cpu().numpy()
                    #     io.plotAbsAngle(c, 'p_hist')
                    p_hist.add(-1, p_now)
                    # if p_hist.size()[2] > 1000:
                    #     c = p_hist[0, 0].cpu().numpy()
                    #     io.plotAbsAngle(c, '(p_hist - p_now)')

                    v_now = p_hist.add(momentum, v_hist)
                    # if v_now.size()[2] > 1000:
                    #     c = v_now[0,0].cpu().numpy()
                    #     io.plotAbsAngle(c,'momentum update')
                    v_history.append(v_now)
                    param_state['v_history'] = v_history
                    if nesterov:
                        d_p = d_p.add(momentum, v_now)
                    else:
                        d_p = d_p.add(v_now)
                elif momentum != 0 and len(history) < T:
                    if 'v_history' not in param_state:
                        tmp = deque()
                        z = p_now.clone().fill_(0+0j)
                        tmp.append(z)
                        param_state['v_history'] = tmp
                    else:
                        v_history = param_state['v_history']
                        z = p_now.clone().fill_(0+0j)
                        v_history.append(z)
                        param_state['v_history'] = v_history
                else:
                    pass
                group['history'] = history

                p.data.add_(-group['lr'], d_p)
        self.step_in_iteration += 1

        return loss
