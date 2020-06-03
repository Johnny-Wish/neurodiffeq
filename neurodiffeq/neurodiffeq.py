import torch
import torch.autograd as autograd


def diff(x, t, order=1):
    """The derivative of a variable with respect to another.

    :param x: The :math:`x` in :math:`\\displaystyle\\frac{\\partial x}{\\partial t}`.
    :type x: `torch.tensor`
    :param t: The :math:`t` in :math:`\\displaystyle\\frac{\\partial x}{\\partial t}`.
    :type t: `torch.tensor`
    :param order: The order of the derivative, defaults to 1.
    :type order: int
    :returns: The derivative.
    :rtype: `torch.tensor`
    """
    ones = torch.ones_like(x)
    der, = autograd.grad(x, t, create_graph=True, grad_outputs=ones)
    for i in range(1, order):
        ones = torch.ones_like(der)
        der, = autograd.grad(der, t, create_graph=True, grad_outputs=ones)
    return der


class CovarianceLoss:
    def __init__(self, t_scale):
        self.t_scale = t_scale

    def __call__(self, Fs, zeros, ts):
        covariance = torch.exp((ts - ts.transpose()) ** 2 / self.t_scale ** 2)
        return torch.chain_matmul(Fs.transpose(), covariance, Fs)
