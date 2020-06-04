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


class SpatiallyWeightedLoss:
    def __init__(self, sq_dist_func, x_scale):
        self.sq_dist_func = sq_dist_func
        self.x_scale = x_scale

    def __call__(self, Fs, zeros, *coordinates):
        return torch.mean((Fs ** 2) * torch.exp(- self.sq_dist_func(*coordinates) / self.x_scale ** 2))
