import torch
import torch.nn as nn
import numpy as np
import copy
from py_DTSTDP import pytorch_STDP as py_STDP
from py_DTSTDP import utils
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import entropy
from datetime import datetime


project_folder, data_folder, figure_folder = utils.get_subfolders(__file__)

N = 100
T = 10000
nbins = 25
f_max = 1
network = py_STDP.RIPLayer(N, N, f_max, True)
x_polar = (f_max * torch.ones(1, N), torch.rand(1, N))
# plt.hist(x_polar[1].detach())

hist_vects_1 = list()
entropy_1 = list()
x = py_STDP.polar_to_cart(x_polar)
for i in range(T+1):
    if i % 100 == 0:
        print(i)
    xp = py_STDP.cart_to_polar(x)
    hist_vect = np.histogram(xp[1].detach(), np.linspace(0, 1, nbins + 1), density=True)[0]
    p = hist_vect / sum(hist_vect)
    entropy_1.append(entropy(p))
    # hist_vect = plt.hist(xp[1].detach(), np.linspace(0, 1, nbins + 1), density=True)
    hist_vects_1.append(torch.tensor(hist_vect).unsqueeze(0))
    x = network.forward(x)
hist_hist_1 = torch.cat(hist_vects_1, dim=0).transpose(0, 1)
# plt.clf()
# plt.subplot(2,1,1)

hist_vects_2 = list()
entropy_2 = list()
x = py_STDP.polar_to_cart(x_polar)
for i in range(T+1):
    if i % 100 == 0:
        print(i)
    xp = py_STDP.cart_to_polar(x)
    hist_vect = np.histogram(xp[1].detach(), np.linspace(0, 1, nbins + 1), density=True)[0]
    p = hist_vect / sum(hist_vect)
    entropy_2.append(entropy(p))
    # hist_vect = plt.hist(xp[1].detach(), np.linspace(0, 1, nbins + 1), density=True)
    hist_vects_2.append(torch.tensor(hist_vect).unsqueeze(0))
    y = network.forward(x)
    network.rip_learn(x, y, 4, stdp_weight=100, hebb_weight=10)
    x = y
hist_hist_2 = torch.cat(hist_vects_2, dim=0).transpose(0, 1)
# plt.subplot(2,1,1)
# min_density = torch.min(torch.min(hist_hist_1), torch.min(hist_hist_2))
# max_density = torch.max(torch.max(hist_hist_1), torch.max(hist_hist_2))
min_density = 0
max_density = 3.5

figure_1, ax = plt.subplots(3, 1, figsize=(8,8), sharex='all')

im0 = ax[0].imshow(hist_hist_1, extent=[1,T,0,1], aspect='auto', vmin=min_density, vmax=max_density)
divider0 = make_axes_locatable(ax[0])
cax0 = divider0.append_axes("right", size="2%", pad=0.05)
plt.colorbar(im0, cax=cax0)
# ax[0].set_aspect(250)
ax[0].set_title('no learning')
# ax[0].set_xlabel('Time step')
ax[0].set_ylabel('Phase offset')

org_hist_2 = utils.orient_img(hist_hist_2.numpy())

# print(hist_hist_2.shape)
im1 = ax[1].imshow(org_hist_2, extent=[1,T,0,1], aspect='auto', vmin=min_density, vmax=max_density)
divider1 = make_axes_locatable(ax[1])
cax1 = divider1.append_axes("right", size="2%", pad=0.05)
plt.colorbar(im1, cax=cax1)
# ax[1].set_aspect(250)
ax[1].set_title('RIP learning')
# ax[1].set_xlabel('Time step')
ax[1].set_ylabel('Phase offset')

# print(len(entropy_2))
# print(hist_hist_2.shape)

ax[2].plot(list(range(1, T+1)), entropy_1[:-1], label='no learning')
ax[2].plot(list(range(1, T+1)), entropy_2[:-1], label='RIP learning')
divider2 = make_axes_locatable(ax[2])
cax2 = divider2.append_axes("right", size="2%", pad=0.05)
cax2.set_axis_off()
ax[2].set_title('Entropy of instantaneous phase distribution')
ax[2].set_xlabel('Time step')
ax[2].set_ylabel('Entropy')
ax[2].set_ylim([0, 3.5])
ax[2].legend(loc='lower left')

plt.suptitle('Evolution of phase distribution')
filename = 'synchronization_1_[{}].eps'.format(datetime.now().strftime('%Y_%m_%d--%H_%M_%S'))
plt.savefig(os.path.join(figure_folder, filename), pad_inches=0)
plt.show()

# def random_vector(f_max, N):
#     rate = f_max * torch.ones(1, N, requires_grad=True)
#     phase = torch.rand(1, N, requires_grad=True)
#     rate[torch.rand_like(rate) < 0.5] = 0.0
#     vector = (rate, phase)
#     return pytorch_STDP.polar_to_cart(vector)
#
#
#
# # x_polar = (f_max * torch.ones(1, N), torch.rand(1, N))
#
# dataset = list()
# for i in range(10):
#     dataset.append(random_vector(1, 32))
#
# ripcoder = RIPCoder()
# optimizer = torch.optim.SGD(ripcoder.parameters(), lr=0.1)
# mse = nn.MSELoss()
#
# std_hist = list()
# error_hist = list()
# entropy_hist = list()
# for i in range(10000):
#     error = torch.zeros(1, requires_grad=True)
#     std = torch.zeros(1)
#     phases = list()
#     for j in range(len(dataset)):
#         x = dataset[j]
#         x_hat, code = ripcoder.forward(x)
#         code_polar = pytorch_STDP.cart_to_polar(code)
#         phases = phases + list(code_polar[1].detach().numpy())
#         std = std + torch.std(code_polar[1])
#         # print(std)
#         error = error + mse.forward(x_hat[0], x[0]) # + mse.forward(x_hat[1], x[1])
#         # print(error.shape)
#     error = error / len(dataset)
#     std = std / len(dataset)
#     phase_p = np.histogram(phases, np.linspace(0, 1, 25), density=True)[0]
#     phase_p = phase_p / np.sum(phase_p)
#     phase_entropy = entropy(phase_p)
#     entropy_hist.append(phase_entropy)
#     if i % 100 == 0:
#         print('{} - ENTROPY:{:.5f}, STD:{:.5f}, ERROR:{:.5f}'.format(i, phase_entropy, std.item(), error.item()))
#     std_hist.append(std.item())
#     error_hist.append(error.item())
#     optimizer.zero_grad()
#     error.backward(retain_graph=True)
#     optimizer.step()
#
#     # if i % 10 == 0:
#     #     print(i, error.detach().item())
#
#
# phases = list()
# for j in range(len(dataset)):
#     x = dataset[j]
#     x_hat, code = ripcoder.forward(x)
#     code_polar = py_STDP.cart_to_polar(code)
#     phases
#     print(code_polar)
#
# f1 = plt.figure()
# plt.subplot(3,1,1)
# plt.plot(error_hist)
# plt.subplot(3,1,2)
# plt.plot(std_hist)
# plt.subplot(3,1,3)
# plt.plot(entropy_hist)
# plt.show()


if False:
    N = 1000
    L = 100
    f_max = 2
    network = RIPNetwork([N] * L, f_max, True)

    x_polar = (f_max*torch.ones(1, N), torch.rand(1, N))
    x = py_STDP.polar_to_cart(x_polar)

    y = network.forward(x)
    f1 = plt.figure()
    hist_vects = list()
    for layer in network.module_list:
        hist_vect = plt.hist(layer.state[1].detach(), np.linspace(0, 1, 25+1), density=True)
        hist_vects.append(torch.tensor(hist_vect[0]).unsqueeze(0))
        # print(hist_vect[0])
        # plt.plot(hist_vect[0])
        # print(torch.std(layer.state[1].detach()))
    hist_hist = torch.cat(hist_vects, dim=0).transpose(0, 1)
    plt.clf()
    plt.imshow(hist_hist)

    # f1 = plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.hist(x_polar[1], np.linspace(0, 1, 25))
    # print(x[1])
    #



    layer = RIPLayer(N, N, f_max, True, 0.5)
    z = layer.forward(x)
    z_polar_original = py_STDP.cart_to_polar(z)
    # plt.subplot(1, 2, 1)

    # f2 = plt.figure()
    # weights = layer.affine.linear.weight.data.view(-1).numpy()
    # plt.hist(weights)

    f2 = plt.figure()

    hist_vects = list()
    for i in range(100):
        z = layer.forward(x)
        hist_vect = plt.hist(layer.state[1].detach(), np.linspace(0, 1, 25 + 1), density=True)
        hist_vects.append(torch.tensor(hist_vect[0]).unsqueeze(0))
        layer.rip_learn(x, z, 3, hebb_weight=1)
        # z_polar = py_STDP.cart_to_polar(z)
        # plt.hist(z[1].detach(), np.linspace(0, 1, 25))
    hist_hist = torch.cat(hist_vects, dim=0).transpose(0, 1)
    plt.clf()
    plt.imshow(hist_hist)

    f3 = plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.hist(x_polar[1], np.linspace(0, 1, 25))
    # plt.subplot(1, 2, 2)

    plt.hist(z_polar_original[1].detach(), np.linspace(0, 1, 25+1))
    z_polar = py_STDP.cart_to_polar(z)
    plt.hist(z_polar[1].detach(), np.linspace(0, 1, 25+1))



    plt.show()

