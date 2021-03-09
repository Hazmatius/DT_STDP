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


data_folder = '/Users/alexbaranski/PycharmProjects/DT_STDP/data/synchronization_experiment_2'
n_experiments = 100

N = 1000
T = 500
nbins = 25
f_max = 1

for n in range(n_experiments):
    network = RIPLayer(N, N, f_max, True)
    x_polar = (f_max * torch.ones(1, N), torch.rand(1, N))

    hist_vects_2 = list()
    entropy_2 = list()
    x = pytorch_STDP.polar_to_cart(x_polar)
    for i in range(T+1):
        if i % 100 == 0:
            print(i)
        xp = pytorch_STDP.cart_to_polar(x)
        hist_vect = np.histogram(xp[1].detach(), np.linspace(0, 1, nbins + 1), density=True)[0]
        p = hist_vect / sum(hist_vect)
        entropy_2.append(entropy(p))
        # hist_vect = plt.hist(xp[1].detach(), np.linspace(0, 1, nbins + 1), density=True)
        hist_vects_2.append(torch.tensor(hist_vect).unsqueeze(0))
        y = network.forward(x)
        network.rip_learn(x, y, 4, stdp_weight=100, hebb_weight=10)
        x = y
    hist_hist_2 = torch.cat(hist_vects_2, dim=0).transpose(0, 1)