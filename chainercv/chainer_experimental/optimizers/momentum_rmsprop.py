import numpy

from chainer import backend
from chainer.backends import cuda
from chainer import optimizer
from chainer import types


if types.TYPE_CHECKING:
    import typing_extensions as tpe

    class MomentumRMSpropHyperparameter(tpe.Protocol):
        """Protocol class for hyperparameter of MomentumRMSprop.
        This is only for PEP 544 compliant static type checkers.
        """
        lr = None  # type: float
        alpha = None  # type: float
        momentum = None  # type: float
        eps = None  # type: float
        eps_inside_sqrt = None  # type: bool


_default_hyperparam = optimizer.Hyperparameter()  # type: MomentumRMSpropHyperparameter # NOQA
_default_hyperparam.lr = 0.01
_default_hyperparam.alpha = 0.99
_default_hyperparam.momentum = 0.9
_default_hyperparam.eps = 1e-8
_default_hyperparam.eps_inside_sqrt = False


class MomentumRMSpropRule(optimizer.UpdateRule):

    """Update rule for MomentumRMSprop.
    """

    def __init__(self, parent_hyperparam=None, lr=None, alpha=None,
                 momentum=None, eps=None, eps_inside_sqrt=None):
        super(MomentumRMSpropRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if lr is not None:
            self.hyperparam.lr = lr
        if alpha is not None:
            self.hyperparam.alpha = alpha
        if momentum is not None:
            self.hyperparam.momentum = momentum
        if eps is not None:
            self.hyperparam.eps = eps
        if eps_inside_sqrt is not None:
            self.hyperparam.eps_inside_sqrt = eps_inside_sqrt

    def init_state(self, param):
        xp = backend.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state['ms'] = xp.zeros_like(param.data)
            self.state['v'] = xp.zeros_like(param.data)

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        eps = grad.dtype.type(hp.eps)
        if hp.eps != 0 and eps == 0:
            raise ValueError(
                'eps of MomentumRMSprop optimizer is too small for {} ({})'.format(
                    grad.dtype.name, hp.eps))
        ms = self.state['ms']
        v = self.state['v']

        ms *= hp.alpha
        ms += (1 - hp.alpha) * grad * grad
        if hp.eps_inside_sqrt:
            denom = numpy.sqrt(ms + eps)
        else:
            denom = numpy.sqrt(ms) + eps
        v *= hp.momentum
        v -= hp.lr * grad / denom
        param.data += v

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        eps = grad.dtype.type(hp.eps)
        if eps == 0:
            raise ValueError(
                'eps of MomentumRMSprop optimizer is too small for {} ({})'.format(
                    grad.dtype.name, hp.eps))
        if hp.eps_inside_sqrt:
            denom = 'sqrt(ms + eps)'
        else:
            denom = 'sqrt(ms) + eps'
        kernel = cuda.elementwise(
            'T grad, T lr, T alpha, T momentum, T eps',
            'T param, T ms, T v',
            '''ms = alpha * ms + (1 - alpha) * grad * grad;
               v = momentum * v - lr * grad / ({});
               param += v;'''.format(denom),
            'momentum_rmsprop')
        kernel(grad, self.hyperparam.lr, self.hyperparam.alpha,
               self.hyperparam.momentum, eps, param.data, self.state['ms'],
               self.state['v'])


class MomentumRMSprop(optimizer.GradientMethod):

    """MomentumRMSprop optimizer.
    """

    def __init__(self, lr=_default_hyperparam.lr,
                 alpha=_default_hyperparam.alpha,
                 momentum=_default_hyperparam.momentum,
                 eps=_default_hyperparam.eps,
                 eps_inside_sqrt=_default_hyperparam.eps_inside_sqrt):
        super(MomentumRMSprop, self).__init__()
        self.hyperparam.lr = lr
        self.hyperparam.alpha = alpha
        self.hyperparam.momentum = momentum
        self.hyperparam.eps = eps
        self.hyperparam.eps_inside_sqrt = eps_inside_sqrt

    lr = optimizer.HyperparameterProxy('lr')
    alpha = optimizer.HyperparameterProxy('alpha')
    momentum = optimizer.HyperparameterProxy('momentum')
    eps = optimizer.HyperparameterProxy('eps')
    eps_inside_sqrt = optimizer.HyperparameterProxy('eps_inside_sqrt')

    def create_update_rule(self):
        return MomentumRMSpropRule(self.hyperparam)
