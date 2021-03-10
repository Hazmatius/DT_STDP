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


def run_sim(arg_dict):
    N = arg_dict['N']
    f_max = arg_dict['f_max']
    T = arg_dict['T']
    nbins = arg_dict['nbins']
    homeostasis = arg_dict['homeostasis']
    sparsity = arg_dict['sparsity']
    a_param = arg_dict['a_param']
    stdp_weight = arg_dict['stdp_weight']
    hebb_weight = arg_dict['hebb_weight']

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
    return dist_hist, entropy_traj


def run_sims(main_data_folder, num_sims, arg_dict):
    args = inspect.getfullargspec(run_sim)
    arg_strings = ['{}={}'.format(arg, eval(arg)) for arg in args]
    desc_string = ','.join(arg_strings)
    data_folder = os.path.join(main_data_folder, desc_string)
    os.mkdir(data_folder)

    for sim_index in range(num_sims):
        dist_hist, entropy_traj = run_sim(arg_dict)
        data = {
            'dist_hist': dist_hist,
            'entropy_traj': entropy_traj
        }
        filename = '{}_[{}].pkl'.format(sim_index, desc_string)
        with open(os.path.join(data_folder, filename), 'wb') as f:
            pickle.dump(data, f)


