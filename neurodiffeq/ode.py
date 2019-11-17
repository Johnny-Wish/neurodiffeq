import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from .networks import FCNN
from copy import deepcopy


def _nn_output(net, t, ith=0):
    output = net(t)
    return output[:, ith].reshape(t.shape)


def _dependant_variables(net, conditions, t):
    return [con.enforce(net, t) for con in conditions]


class NoCondition:
    """An condition class that does not impose any initial/boundary conditions

    :param ith: Impose this condition on the i-th output unit of the neural network
    :type ith: int
    """
    def __init__(self, ith=0):
        """Initializer method
        """
        self.ith = ith

    def enforce(self, net, t):
        r"""Return the raw input of neural network.

        .. note::
            `enforce` is meant to be called by the function `solve` and `solve_system`.
        """
        return _nn_output(net, t, self.ith)

    def set_impose_on(self, ith):
        r"""Impose this condition on the i-th output unit of the neural network

        .. note::
            `set_impose_on` is meant to be called by the function `solve` and `solve_system`.
        """
        self.ith = ith


class IVP:
    """An initial value problem.
        For Dirichlet condition, we are solving :math:`x(t)` given :math:`x(t)\\bigg|_{t = t_0} = x_0`.
        For Neumann condition, we are solving :math:`x(t)` given :math:`\\displaystyle\\frac{\\partial x}{\\partial t}\\bigg|_{t = t_0} = x_0'`.

    :param t_0: The initial time.
    :type t_0: float
    :param x_0: The initial value of :math:x. :math:`x(t)\\bigg|_{t = t_0} = x_0`.
    :type x_0: float
    :param x_0_prime: The inital derivative of :math:`x` wrt :math:`t`. :math:`\\displaystyle\\frac{\\partial x}{\\partial t}\\bigg|_{t = t_0} = x_0'`, defaults to None.
    :type x_0_prime: float, optional
    :param ith: Impose this condition on the i-th output unit of the neural network
    :type ith: int
    """
    def __init__(self, t_0, x_0, x_0_prime=None, ith=0):
        """Initializer method
        """
        self.t_0, self.x_0, self.x_0_prime = t_0, x_0, x_0_prime
        self.ith = ith

    def enforce(self, net, t):
        r"""Enforce the output of a neural network to satisfy the initial condition.

        :param net: The neural network that approximates the ODE.
        :type net: `torch.nn.Module`
        :param t: The points where the neural network output is evaluated.
        :type t: `torch.tensor`
        :return: The modified output which now satisfies the initial condition.
        :rtype: `torch.tensor`

        .. note::
            `enforce` is meant to be called by the function `solve` and `solve_system`.
        """
        x = _nn_output(net, t, self.ith)
        if self.x_0_prime:
            return self.x_0 + (t-self.t_0)*self.x_0_prime + ( (1-torch.exp(-t+self.t_0))**2 )*x
        else:
            return self.x_0 + (1-torch.exp(-t+self.t_0))*x

    def set_impose_on(self, ith):
        r"""Impose this condition on the i-th output unit of the neural network

        .. note::
            `set_impose_on` is meant to be called by the function `solve` and `solve_system`.
        """
        self.ith = ith


class DirichletBVP:
    """A two-point Dirichlet boundary condition.
        We are solving :math:`x(t)` given :math:`x(t)\\bigg|_{t = t_0} = x_0` and :math:`x(t)\\bigg|_{t = t_1} = x_1`.

    :param t_0: The initial time.
    :type t_0: float
    :param t_1: The final time.
    :type t_1: float
    :param x_0: The initial value of :math:x. :math:`x(t)\\bigg|_{t = t_0} = x_0`.
    :type x_0: float
    :param x_1: The initial value of :math:x. :math:`x(t)\\bigg|_{t = t_1} = x_1`.
    :type x_1: float
    :param ith: Impose this condition on the i-th output unit of the neural network
    :type ith: int
    """
    def __init__(self, t_0, x_0, t_1, x_1, ith=0):
        """Initializer method
        """
        self.t_0, self.x_0, self.t_1, self.x_1 = t_0, x_0, t_1, x_1
        self.ith = ith

    def enforce(self, net, t):
        r"""Enforce the output of a neural network to satisfy the boundary condition.

        :param net: The neural network that approximates the ODE.
        :type net: `torch.nn.Module`
        :param t: The points where the neural network output is evaluated.
        :type t: `torch.tensor`
        :return: The modified output which now satisfies the boundary condition.
        :rtype: `torch.tensor`


        .. note::
            `enforce` is meant to be called by the function `solve` and `solve_system`.
        """
        x = net(t)
        t_tilde = (t-self.t_0) / (self.t_1-self.t_0)
        return self.x_0*(1-t_tilde) + self.x_1*t_tilde + (1-torch.exp((1-t_tilde)*t_tilde))*x

    def set_impose_on(self, ith):
        r"""Impose this condition on the i-th output unit of the neural network

        .. note::
            `set_impose_on` is meant to be called by the function `solve` and `solve_system`.
        """
        self.ith = ith


class ExampleGenerator:
    """An example generator for generating 1-D training points.

    :param size: The number of points to generate each time `get_examples` is called.
    :type size: int
    :param t_min: The lower bound of the 1-D points generated, defaults to 0.0.
    :type t_min: float, optional
    :param t_max: The upper boound of the 1-D points generated, defaults to 1.0.
    :type t_max: float, optional
    :param method: The distribution of the 1-D points generated.
        If set to 'uniform', the points will be drew from a uniform distribution Unif(t_min, t_max).
        If set to 'equally-spaced', the points will be fixed to a set of linearly-spaced points that go from t_min to t_max.
        If set to 'equally-spaced-noisy', a normal noise will be added to the previously mentioned set of points.
        If set to 'log-spaced', the points will be fixed to a set of log-spaced points that go from t_min to t_max.
        If set to 'log-spaced-noisy', a normal noise will be added to the previously mentioned set of points, defaults to 'uniform'.
    :type method: str, optional
    :raises ValueError: When provided with an unknown method.
    """
    def __init__(self, size, t_min=0.0, t_max=1.0, method='uniform'):
        r"""Initializer method

        .. note::
            A instance method `get_examples` is dynamically created to generate 1-D training points. It will be called by the function `solve` and `solve_system`.
        """
        self.size = size
        self.t_min, self.t_max = t_min, t_max
        if method == 'uniform':
            self.examples = torch.zeros(self.size, requires_grad=True)
            self.get_examples = lambda: self.examples + torch.rand(self.size)*(self.t_max-self.t_min) + self.t_min
        elif method == 'equally-spaced':
            self.examples = torch.linspace(self.t_min, self.t_max, self.size, requires_grad=True)
            self.get_examples = lambda: self.examples
        elif method == 'equally-spaced-noisy':
            self.examples = torch.linspace(self.t_min, self.t_max, self.size, requires_grad=True)
            self.noise_mean = torch.zeros(self.size)
            self.noise_std  = torch.ones(self.size) * ( (t_max-t_min)/size ) / 4.0
            self.get_examples = lambda: self.examples + torch.normal(mean=self.noise_mean, std=self.noise_std)
        elif method == 'log-spaced':
            self.examples = torch.logspace(self.t_min, self.t_max, self.size, requires_grad=True)
            self.get_examples = lambda: self.examples
        elif method == 'log-spaced-noisy':
            self.examples = torch.logspace(self.t_min, self.t_max, self.size, requires_grad=True)
            self.noise_mean = torch.zeros(self.size)
            self.noise_std = torch.ones(self.size) * ((t_max - t_min) / size) / 4.0
            self.get_examples = lambda: self.examples + torch.normal(mean=self.noise_mean, std=self.noise_std)
        else:
            raise ValueError(f'Unknown method: {method}')


class Monitor:
    """A monitor for checking the status of the neural network during training.

    :param t_min: The lower bound of time domain that we want to monitor.
    :type t_min: float
    :param t_max: The upper bound of time domain that we want to monitor.
    :type t_max: float
    :param check_every: The frequency of checking the neural network represented by the number of epochs between two checks, defaults to 100.
    :type check_every: int, optional
    """
    def __init__(self, t_min, t_max, check_every=100):
        """Initializer method
        """
        self.check_every = check_every
        self.fig = plt.figure(figsize=(20, 8))
        self.ax1 = self.fig.add_subplot(121)
        self.ax2 = self.fig.add_subplot(122)
        # input for plotting
        self.ts_plt = np.linspace(t_min, t_max, 100)
        # input for neural network
        self.ts_ann = torch.linspace(t_min, t_max, 100, requires_grad=True).reshape((-1, 1))

    def check(self, net, conditions, loss_history):
        r"""Draw 2 plots: One shows the shape of the current solution. The other shows the history training loss and validation loss.

        :param net: The neural network that approximates the ODE (system).
        :type net: `torch.nn.Module`
        :param conditions: The initial/boundary conditions of the ODE (system).
        :type conditions: list[`neurodiffeq.ode.IVP` or `neurodiffeq.ode.DirichletBVP` or `neurodiffeq.ode.NoCondition`]
        :param loss_history: The history of training loss and validation loss. The 'train' entry is a list of training loss and 'valid' entry is a list of validation loss.
        :type loss_history: dict['train': list[float], 'valid': list[float]]

        .. note::
            `check` is meant to be called by the function `solve` and `solve_system`.
        """
        n_dependent = len(conditions)

        vs = _dependant_variables(net, conditions, self.ts_ann)
        vs = [v.detach().numpy() for v in vs]

        self.ax1.clear()
        for i in range(n_dependent):
            self.ax1.plot(self.ts_plt, vs[i], label=f'variable {i}')
        self.ax1.legend()
        self.ax1.set_title('solutions')

        self.ax2.clear()
        self.ax2.plot(loss_history['train'], label='training loss')
        self.ax2.plot(loss_history['valid'], label='validation loss')
        self.ax2.set_title('loss during training')
        self.ax2.set_ylabel('loss')
        self.ax2.set_xlabel('epochs')
        self.ax2.set_yscale('log')
        self.ax2.legend()

        self.fig.canvas.draw()


def solve(
        ode, condition, t_min, t_max,
        net=None, train_generator=None, shuffle=True, valid_generator=None,
        optimizer=None, criterion=None, batch_size=16,
        max_epochs=1000,
        monitor=None, return_internal=False,
        return_best=False
):
    """Train a neural network to solve an ODE.

    :param ode: The ODE to solve. If the ODE is :math:`F(x, t) = 0` where :math:`x` is the dependent variable and :math:`t` is the independent variable,
        then `ode` should be a function that maps :math:`(x, t)` to :math:`F(x, t)`.
    :type ode: function
    :param condition: The initial/boundary condition.
    :type condition: `neurodiffeq.ode.IVP` or `neurodiffeq.ode.DirichletBVP` or `neurodiffeq.ode.NoCondition`
    :param net: The neural network used to approximate the solution, defaults to None.
    :type net: `torch.nn.Module`, optional
    :param t_min: The lower bound of the domain (t) on which the ODE is solved.
    :type t_min: float
    :param t_max: The upper bound of the domain (t) on which the ODE is solved.
    :type t_max: float
    :param train_generator: The example generator to generate 1-D training points, default to None.
    :type train_generator: `neurodiffeq.ode.ExampleGenerator`, optional
    :param shuffle: Whether to shuffle the training examples every epoch, defaults to True.
    :type shuffle: bool, optional
    :param valid_generator: The example generator to generate 1-D validation points, default to None.
    :type valid_generator: `neurodiffeq.ode.ExampleGenerator`, optional
    :param optimizer: The optimization method to use for training, defaults to None.
    :type optimizer: `torch.optim.Optimizer`, optional
    :param criterion: The loss function to use for training, defaults to None.
    :type criterion: `torch.nn.modules.loss._Loss`, optional
    :param batch_size: The size of the mini-batch to use, defaults to 16.
    :type batch_size: int, optional
    :param max_epochs: The maximum number of epochs to train, defaults to 1000.
    :type max_epochs: int, optional
    :param monitor: The monitor to check the status of nerual network during training, defaults to None.
    :type monitor: `neurodiffeq.ode.Monitor`, optional
    :param return_internal: Whether to return the net, conditions, training generator, validation generator, optimizer and loss function, defaults to False.
    :type return_internal: bool, optional
    :param return_best: Whether to return the net that achieved the lowest validation loss, defaults to False.
    :type return_best: bool, optional
    :return: The solution of the ODE. The history of training loss and validation loss.
        Optionally, the net, conditions, training generator, validation generator, optimizer and loss function.
    :rtype: tuple[`neurodiffeq.ode.Solution`, dict]; or tuple[`neurodiffeq.ode.Solution`, dict, dict]
    """
    return solve_system(
        ode_system=lambda x, t: [ode(x, t)], conditions=[condition],
        t_min=t_min, t_max=t_max, net=net,
        train_generator=train_generator, shuffle=shuffle, valid_generator=valid_generator,
        optimizer=optimizer, criterion=criterion, batch_size=batch_size,
        max_epochs=max_epochs, monitor=monitor, return_internal=return_internal,
        return_best=return_best
    )


def solve_system(
        ode_system, conditions, t_min, t_max,
        net=None, train_generator=None, shuffle=True, valid_generator=None,
        optimizer=None, criterion=None, batch_size=16,
        max_epochs=1000,
        monitor=None, return_internal=False,
        return_best=False,
):
    """Train a neural network to solve an ODE.

    :param ode_system: The ODE system to solve. If the ODE system consists of equations :math:`F_i(x_1, x_2, ..., x_n, t) = 0` where :math:`x_i` is the dependent i-th variable and :math:`t` is the independent variable,
        then `ode_system` should be a function that maps :math:`(x_1, x_2, ..., x_n, t)` to a list where the i-th entry is :math:`F_i(x_1, x_2, ..., x_n, t)`.
    :type ode_system: function
    :param conditions: The initial/boundary conditions. The ith entry of the conditions is the condition that :math:`x_i` should satisfy.
    :type conditions: list[`neurodiffeq.ode.IVP` or `neurodiffeq.ode.DirichletBVP` or `neurodiffeq.ode.NoCondition`]
    :param t_min: The lower bound of the domain (t) on which the ODE is solved.
    :type t_min: float
    :param t_max: The upper bound of the domain (t) on which the ODE is solved.
    :type t_max: float
    :param net: The neural network used to approximate the solution, defaults to None.
    :type net: `torch.nn.Module`, optional
    :param train_generator: The example generator to generate 1-D training points, default to None.
    :type train_generator: `neurodiffeq.ode.ExampleGenerator`, optional
    :param shuffle: Whether to shuffle the training examples every epoch, defaults to True.
    :type shuffle: bool, optional
    :param valid_generator: The example generator to generate 1-D validation points, default to None.
    :type valid_generator: `neurodiffeq.ode.ExampleGenerator`, optional
    :param optimizer: The optimization method to use for training, defaults to None.
    :type optimizer: `torch.optim.Optimizer`, optional
    :param criterion: The loss function to use for training, defaults to None.
    :type criterion: `torch.nn.modules.loss._Loss`, optional
    :param batch_size: The size of the mini-batch to use, defaults to 16.
    :type batch_size: int, optional
    :param max_epochs: The maximum number of epochs to train, defaults to 1000.
    :type max_epochs: int, optional
    :param monitor: The monitor to check the status of nerual network during training, defaults to None.
    :type monitor: `neurodiffeq.ode.Monitor`, optional
    :param return_internal: Whether to return the net, conditions, training generator, validation generator, optimizer and loss function, defaults to False.
    :type return_internal: bool, optional
    :param return_best: Whether to return the net that achieved the lowest validation loss, defaults to False.
    :type return_best: bool, optional
    :return: The solution of the ODE. The history of training loss and validation loss.
        Optionally, the net, conditions, training generator, validation generator, optimizer and loss function.
    :rtype: tuple[`neurodiffeq.ode.Solution`, dict]; or tuple[`neurodiffeq.ode.Solution`, dict, dict]
    """

    if not net:
        net = FCNN(n_input_units=1, n_output_units=len(conditions), n_hidden_units=32, n_hidden_layers=1)
    if not train_generator:
        train_generator = ExampleGenerator(32, t_min, t_max, method='equally-spaced-noisy')
    if not valid_generator:
        valid_generator = ExampleGenerator(32, t_min, t_max, method='equally-spaced')
    if not optimizer:
        optimizer = optim.Adam(net.parameters(), lr=0.001)
    if not criterion:
        criterion = nn.MSELoss()

    for ith, con in enumerate(conditions):
        con.set_impose_on(ith)

    if return_internal:
        internal = {
            'net': net,
            'conditions': conditions,
            'train_generator': train_generator,
            'valid_generator': valid_generator,
            'optimizer': optimizer,
            'criterion': criterion
        }

    n_examples_train = train_generator.size
    n_examples_valid = valid_generator.size
    train_zeros = torch.zeros(batch_size).reshape((-1, 1))
    valid_zeros = torch.zeros(n_examples_valid).reshape((-1, 1))

    loss_history = {'train': [], 'valid': []}
    valid_loss_epoch_min = np.inf
    solution_min = None

    for epoch in range(max_epochs):
        train_loss_epoch = 0.0

        train_examples = train_generator.get_examples()
        train_examples = train_examples.reshape(n_examples_train, 1)
        idx = np.random.permutation(n_examples_train) if shuffle else np.arange(n_examples_train)
        batch_start, batch_end = 0, batch_size
        while batch_start < n_examples_train:

            if batch_end >= n_examples_train:
                batch_end = n_examples_train
            batch_idx = idx[batch_start:batch_end]
            ts = train_examples[batch_idx]

            vs = _dependant_variables(net, conditions, ts)

            fvts = ode_system(*vs, ts)
            loss = sum(criterion(fvt, train_zeros) for fvt in fvts)
            train_loss_epoch += loss.item() * (batch_end-batch_start)/n_examples_train # assume the loss is a mean over all examples

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_start += batch_size
            batch_end += batch_size

        loss_history['train'].append(train_loss_epoch)

        # calculate the validation loss
        ts = valid_generator.get_examples().reshape(n_examples_valid, 1)
        vs = _dependant_variables(net, conditions, ts)
        fvts = ode_system(*vs, ts)
        valid_loss_epoch = sum(criterion(fvt, valid_zeros) for fvt in fvts)
        valid_loss_epoch = valid_loss_epoch.item()

        loss_history['valid'].append(valid_loss_epoch)

        if monitor and epoch % monitor.check_every == 0:
            monitor.check(net, conditions, loss_history)

        if return_best and valid_loss_epoch < valid_loss_epoch_min:
            valid_loss_epoch_min = valid_loss_epoch
            solution_min = Solution(net, conditions)

    if return_best:
        solution = solution_min
    else:
        solution = Solution(net, conditions)

    if return_internal:
        return solution, loss_history, internal
    else:
        return solution, loss_history


class Solution:
    """A solution to an ODE (system)

    :param net: The neural network that approximates the ODE.
    :type net: `torch.nn.Module`
    :param conditions: The initial/boundary conditions of the ODE (system).
    :type conditions: list[`neurodiffeq.ode.IVP` or `neurodiffeq.ode.DirichletBVP` or `neurodiffeq.ode.NoCondition`]
    """
    def __init__(self, net, conditions):
        """Initializer method
        """
        self.net = deepcopy(net)
        self.conditions = deepcopy(conditions)

    def __call__(self, ts, as_type='tf'):
        """Evaluate the solution at certain points.

        :param ts: the points on which the dependent variables are evaluated.
        :type ts: `torch.tensor` or sequence of number
        :param as_type: Whether the returned value is a `torch.tensor` ('tf') or `numpy.array` ('np').
        :type as_type: str
        :return: dependent variables are evaluated at given points.
        :rtype: list[`torch.tensor` or `numpy.array` (when there is more than one dependent variables)
            `torch.tensor` or `numpy.array` (when there is only one dependent variable)
        """
        if not isinstance(ts, torch.Tensor):
            ts = torch.tensor(ts, dtype=torch.float32)
        original_shape = ts.shape
        ts = ts.reshape(-1, 1)
        if as_type not in ('tf', 'np'):
            raise ValueError("The valid return types are 'tf' and 'np'.")

        vs = _dependant_variables(self.net, self.conditions, ts)
        vs = [v.reshape(original_shape) for v in vs]
        if as_type == 'np':
            vs = [v.detach().numpy() for v in vs]

        return vs if len(vs) > 1 else vs[0]
