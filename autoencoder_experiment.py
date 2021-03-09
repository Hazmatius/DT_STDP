import torch
import torch.nn as nn
import numpy as np
import copy
import pytorch_STDP
import os
from pytorch_STDP import RIPLayer
from pytorch_STDP import RIPNetwork
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy
from scipy.stats import entropy
from datetime import datetime


class RIPCoder(nn.Module):
    def __init__(self, encoder_dims, max_encoder_rate, max_decoder_rate):
        super(RIPCoder, self).__init__()
        decoder_dims = [i for i in reversed(encoder_dims)]
        self.encoder = RIPNetwork(encoder_dims, max_encoder_rate, True)
        self.decoder = RIPNetwork(decoder_dims, max_decoder_rate)

    def forward(self, input):
        code = self.encoder(input)
        return self.decoder(code), code


def random_vector(f_max, N):
    rate = f_max * torch.ones(1, N, requires_grad=True)
    phase = torch.zeros(1, N, requires_grad=True)
    rate[torch.rand_like(rate) < 0.65] = 0.0
    vector = (rate, phase)
    return pytorch_STDP.polar_to_cart(vector)


def jitter(complex_tensor):
    tensor_polar = pytorch_STDP.cart_to_polar(complex_tensor)
    rate = tensor_polar[0]
    phase = tensor_polar[1]
    jitter_tensor = pytorch_STDP.polar_to_cart((rate, phase + torch.rand_like(phase)/20))
    return jitter_tensor


def get_error_history(dataset, add_jitter, encoder_dims, max_encoder_rate, max_decoder_rate):
    ripcoder = RIPCoder(encoder_dims, max_encoder_rate, max_decoder_rate)
    optimizer = torch.optim.SGD(ripcoder.parameters(), lr=0.01)
    mse = nn.MSELoss()

    error_hist = list()

    for i in range(10000):
        error = torch.zeros(1, requires_grad=True)
        for j in range(len(dataset)):
            if add_jitter:
                x = jitter(dataset[j])
            else:
                x = dataset[j]
            # print(x)
            x_hat, code = ripcoder.forward(x)
            x_hat_polar = pytorch_STDP.cart_to_polar(x_hat)
            x_polar = pytorch_STDP.cart_to_polar(x)
            error = error + mse.forward(x_hat_polar[0], x_polar[0])
        if i % 100 == 0:
            print('    {} - ERROR:{:.5f}'.format(i, error.item()))
        error_hist.append(error.item())
        optimizer.zero_grad()
        error.backward(retain_graph=True)
        optimizer.step()
    return error_hist, ripcoder


def get_avg_trajectory(traj_list):
    return sum([np.array(traj) for traj in traj_list]) / len(traj_list)


folder = '/Users/alexbaranski/Desktop/fig_folder/'
without_jitter = list()
with_jitter = list()
without_jitter_models = list()
with_jitter_models = list()

dataset = list()
for i in range(10):
    dataset.append(random_vector(1, 32))

for i in range(3):
    print('no jitter')
    error_hist, ripnet = get_error_history(dataset, False)
    without_jitter.append(error_hist)
    without_jitter_models.append(ripnet)
    # without_jitter.append(get_error_history(dataset, False))
    print('jitter')
    error_hist, ripnet = get_error_history(dataset, True)
    with_jitter.append(error_hist)
    with_jitter_models.append(ripnet)
    # with_jitter.append(get_error_history(dataset, True))