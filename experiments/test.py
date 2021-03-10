import torch
import numpy as np
import os
from py_DTSTDP import pytorch_STDP as py_STDP
from py_DTSTDP import utils
from scipy.stats import entropy
import pathlib
from datetime import datetime
import pickle
from scipy import signal
import matplotlib.pyplot as plt
import inspect


def run_sim(index, arg_dict):
    N = arg_dict['N']
    f_max = arg_dict['f_max']
    T = arg_dict['T']
    nbins = arg_dict['nbins']
    homeostasis = arg_dict['homeostasis']
    sparsity = arg_dict['sparsity']
    a_param = arg_dict['a_param']
    stdp_weight = arg_dict['stdp_weight']
    hebb_weight = arg_dict['hebb_weight']

    args = inspect.getfullargspec(run_sim)
    arg_strings = ['{}={}'.format(arg, eval(arg)) for arg in args]
    desc_string = 'index={},'.format(index) + ','.join(arg_strings)

    network = py_STDP.RIPLayer(N, N, f_max, homeostasis, sparsity)
    x_polar = (f_max * torch.ones(1, N), torch.rand(1, N))

    hist_vects = list()
    entropy_traj = list()
    x = py_STDP.polar_to_cart(x_polar)
    for i in range(T + 1):
        if i % 100 == 0:
            print('   {}'.format(i))
        xp = py_STDP.cart_to_polar(x)
        hist_vect = np.histogram(xp[1].detach(), np.linspace(0, 1, nbins + 1), density=True)[0]
        p = hist_vect / sum(hist_vect)
        entropy_traj.append(entropy(p))
        hist_vects.append(torch.tensor(hist_vect).unsqueeze(0))
        y = network.forward(x)
        network.rip_learn(x, y, a_param, stdp_weight=stdp_weight, hebb_weight=hebb_weight)
        x = y
    dist_hist = torch.cat(hist_vects, dim=0).transpose(0, 1)
    return dist_hist, entropy_traj, desc_string


config_spec = {
    'N': [10, 31, 100, 310, 1000],
    'f_max': [1, 2, 4, 8],
    'T': [1000],
    'nbins': [25],
    'homeostasis': [True, False],
    'sparsity': [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1],
    'a_param': [4],
    'stdp_weight': [100],
    'hebb_weight': [10]
}

all_configs = utils.dict_factor(config_spec)
for config in all_configs:
    print(config)


# N = 1000
# T = 1000
# f_max = 1
# nbins = 25
# a_term = 4
# stdp_weight = 100
# hebb_weight = 10
#
# dist_hist, entropy_traj, desc_string = run_sim(N, f_max, T, nbins)
# org_hist = utils.orient_img(dist_hist.numpy())
# # plt.imshow(org_hist)
# plt.imshow(org_hist, extent=[1,T,0,1], aspect='auto')
# plt.show()
# # print(network.affine.linear.weight.data)