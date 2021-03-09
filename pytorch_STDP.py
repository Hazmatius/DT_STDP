import torch
import torch.nn as nn
import numpy as np
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import matplotlib.pyplot as plt


def modulo(n, m):
    return n - torch.floor(n / m) * m


def cartesian_metric(x, y):
    return torch.sqrt(torch.sum((y[0]-x[0])**2 + (y[1]-x[1])**2)).detach()/torch.numel(x[0])


def phase_metric(x, y):
    xp = cart_to_polar(x)
    yp = cart_to_polar(y)
    return None


def cart_to_polar(input):
    return torch.sqrt(input[0] ** 2 + input[1] ** 2), modulo(torch.atan2(input[1], input[0]) / (2 * np.pi), 1)


def polar_to_cart(input):
    return input[0] * torch.cos(2 * np.pi * input[1]), input[0] * torch.sin(2 * np.pi * input[1])


class RIPNonlinear(nn.Module):
    def __init__(self, f_max):
        super(RIPNonlinear, self).__init__()
        self.f_max = f_max
        self.mod = Modulo()
        self.tanh = nn.Tanh()

    def forward(self, input):
        return self.f_max * self.tanh(input[0] / self.f_max), 2 * np.pi * self.mod(input[1] / (2 * np.pi), 1/input[0])


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, p=0.5):
        super(ComplexLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        # self.linear.weight.data = torch.abs(self.linear.weight.data)
        # inhib = torch.rand(in_features) > p
        # self.linear.weight.data[inhib, :] = -self.linear.weight.data[inhib, :]
        # print(self.linear.weight.data.view(-1))

    def forward(self, input):
        return self.linear(input[0]), self.linear(input[1])


class Modulo(nn.Module):
    def __init__(self):
        super(Modulo, self).__init__()

    def forward(self, n, m):
        m_prime = torch.max(m, torch.tensor(1))
        return modulo(n, m_prime)


class RIPLayer(nn.Module):
    def __init__(self, in_dim, out_dim, f_max, homeostasis=False, bias=False, p=0.5):
        super(RIPLayer, self).__init__()
        self.rip = RIPNonlinear(f_max)
        self.affine = ComplexLinear(in_dim, out_dim, bias, p)
        self.mod = Modulo()
        self.state = None
        self.homeostasis = homeostasis

    def forward(self, input):
        a_cart = self.affine(input)
        a_polar = cart_to_polar(a_cart)
        # z_polar = a_polar
        z_polar = self.rip(a_polar)
        if self.homeostasis:
            z_polar = (torch.ones_like(z_polar[0]), z_polar[1])
        z_cart = polar_to_cart(z_polar)
        self.state = z_polar
        return z_cart

    def rip_learn(self, x, y, a, **kwargs):
        f1, w1 = x
        f2, w2 = y

        if 'stdp_weight' in kwargs:
            stdp_weight = kwargs['stdp_weight']
        else:
            stdp_weight = 1
        if 'hebb_weight' in kwargs:
            hebb_weight = kwargs['hebb_weight']
        else:
            hebb_weight = 1

        # print('------')
        # print(f1.shape)
        # print(w1.shape)
        # print(f2.shape)
        # print(w2.shape)
        # print('------')
        hebb_term = torch.matmul(torch.transpose(f2, 0, 1), f1)
        factor = 8*a*hebb_term/((a+2*hebb_term)**2)
        # print(torch.max(factor))
        phase_diff = torch.transpose(w2, 0, 1) - w1
        stdp_term = 2*torch.floor(phase_diff/2)+1-phase_diff
        stdp_term = stdp_term / torch.norm(stdp_term)
        hebb_term = hebb_term / torch.norm(hebb_term)
        dW = factor*stdp_weight*stdp_term + hebb_weight*hebb_term

        # print(stdp_term.shape)
        # print(hebb_term.shape)
        # print(dW.shape)
        # print(self.affine.linear.weight.data.shape)

        self.affine.linear.weight.data = self.affine.linear.weight.data * 0.999 + dW/100


class RIPNetwork(nn.Module):
    def __init__(self, layer_dims, fmax, homeostasis=False, bias=False, p=0.5):
        super(RIPNetwork, self).__init__()

        self.module_list = list()
        for i in range(len(layer_dims)-1):
            self.module_list.append(RIPLayer(layer_dims[i], layer_dims[i+1], fmax, homeostasis, bias, p))
        self.network = nn.ModuleList(self.module_list)

    def forward(self, input):
        for module in self.network:
            input = module.forward(input)
        return input


# theta = torch.tensor(np.arange(0, 2*np.pi, 0.01))
# radius = torch.ones_like(theta)
# x = radius * torch.cos(theta)
# y = radius * torch.sin(theta)
#
# # print(torch.fmod(torch.tensor(-1.5), torch.tensor(1.0)))
#
# theta_hat = modulo(torch.atan2(y, x), torch.tensor(2*np.pi))
# plt.plot(theta, theta_hat)
# plt.show()
#
#
# mod = Modulo()
#
# x = torch.tensor(-1.5)
# T = torch.tensor(1.0)
# print(mod.forward(x, T))


# mse = nn.MSELoss()
# model = RIPNetwork([10, 10], 5)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
#
# x = (10*torch.ones(1, 100), torch.zeros(1, 100))
# y = (10*torch.ones(1, 90), torch.zeros(1, 90))
#
# x[0][torch.rand(1, 100) < 0.7] = 0
# y[0][torch.rand(1, 90) < 0.7] = 0
# target = torch.rand(1, 10)
# # target = torch.polar(torch.rand(1, 10), torch.rand(1, 10) * 2 * np.pi)
#
# riplayer = RIPLayer(100, 90, 100)
# for i in range(1000):
#     yhat = riplayer.forward(x)
#     if i % 100 == 0:
#         print(cartesian_metric(y, yhat))
#     # print(yhat)
#     riplayer.rip_learn(x, y, 1)
# # print(cartesian_metric(y, riplayer.forward(x)))
# for i in range(torch.numel(y[0])):
#     print('{:+.3f} - {:+.3f}'.format(y[0][0, i].item(), yhat[0][0, i].item()))
#     # print(y[0][0, i].item())
# print(y[0])
# print(yhat[0])
# for i in range(torch.numel(yhat[0])):
#     print(y[0][i], yhat[0][i])


# for i in range(100):
#     output = model.forward(x)
#     loss = mse(output[1], target)
#     optimizer.zero_grad()
#     loss.backward()
#     print(loss)
#     optimizer.step()