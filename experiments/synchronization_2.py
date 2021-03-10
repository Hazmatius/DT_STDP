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


def run_sim(N, f_max, T, nbins):
    network = py_STDP.RIPLayer(N, N, f_max, True)
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
        network.rip_learn(x, y, 4, stdp_weight=100, hebb_weight=10)
        x = y
    dist_hist = torch.cat(hist_vects, dim=0).transpose(0, 1)
    return dist_hist, entropy_traj


project_folder, data_folder, figure_folder = utils.get_subfolders(__file__)


n_experiments = 1000

N = 1000
T = 1000
nbins = 25
f_max = 1

n_data_dicts = list()

for n in range(n_experiments):
    timestamp = datetime.now().strftime('%Y_%m_%d--%H_%M_%S')
    dist_hist, entropy_traj = run_sim(N, f_max, T, nbins)
    data = {'dist_hist': dist_hist, 'entropy': entropy_traj}
    with open(os.path.join(data_folder, '{}.pkl'.format(timestamp)), 'wb') as f:
        pickle.dump(data, f)


data_files = os.listdir(data_folder)
all_val_list = list()
all_flux_list = list()
for data_file in data_files:
    with open(os.path.join(data_folder, data_file), 'rb') as f:
        data = pickle.load(f)
        entropy_traj = data['entropy']
        val_array, min_array, max_array, flux_array = utils.get_minmax_array(entropy_traj, 200)
        all_val_list.append(val_array)
        all_flux_list.append(flux_array)

all_val_array = np.concatenate(all_val_list)
all_flux_array = np.concatenate(all_flux_list)

f2 = plt.figure()
plt.hist2d(all_val_array, all_flux_array)
plt.show()


# f1 = plt.figure()
# plt.plot(entropy_traj, color=[0, 0, 0])
#
# for i in [200]:
#     val_array, min_array, max_array, flux_array = utils.get_minmax_array(entropy_traj, i)
#     plt.plot(flux_array, color=[0, 1, 0])
#     plt.plot(min_array, color=[1, 0, 0])
#     plt.plot(max_array, color=[0, 0, 1])
#
# f2 = plt.figure()
# plt.scatter(val_array, flux_array)
# plt.show()

# fs = 1
# lags = [1, 2, 4, 8, 16, 32, 64, 128, 256]
# f, Pxx_spec = signal.welch(entropy_traj, fs, 'flattop', 1024, scaling='spectrum')
# plt.semilogy(f, np.sqrt(Pxx_spec))
# plt.show()

