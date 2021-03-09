import torch
import torch.nn as nn
import numpy as np
import copy
import pytorch_STDP
import os
from pytorch_STDP import RIPLayer
from pytorch_STDP import RIPNetwork
import matplotlib.pyplot as plt
from datetime import datetime



def random_vector(f_max, p, N):
    rate = f_max * torch.ones(1, N, requires_grad=True)
    phase = torch.zeros(1, N, requires_grad=True)
    rate[torch.rand_like(rate) > p] = 0.0
    vector = (rate, phase)
    return pytorch_STDP.polar_to_cart(vector)


def zero_vector(N):
    rate = torch.zeros(1, N, requires_grad=True)
    phase = torch.zeros(1, N, requires_grad=True)
    vector = (rate, phase)
    return vector


def cat_vects(vector_A, vector_B):
    dim = 1
    vector_C = (torch.cat([vector_A[0], vector_B[0]], dim=dim), torch.cat([vector_A[1], vector_B[1]], dim=dim))
    return vector_C


f_max = 10
density = 0.05
n = 200
c = 10

Z = zero_vector(c)
S = random_vector(f_max, .4, c)
A = random_vector(f_max, density, n)
A_switch = cat_vects(A, S)
# print(A_switch[0].shape)
# print(A[0].shape)
B = random_vector(f_max, density, n)
C = random_vector(f_max, density, n)
C_switch = cat_vects(C, S)
D = random_vector(f_max, density, n)

Az = cat_vects(A, Z)
Bz = cat_vects(B, Z)
Cz = cat_vects(C, Z)
Dz = cat_vects(D, Z)

network = RIPLayer(n+c, n, f_max, homeostasis=False)

iters = 10000
for i in range(iters):
    if i % 100 == 0:
        print('{}%'.format(i/iters * 100))
    network.rip_learn(Az, B, f_max, stdp_weight=1, hebb_weight=1)
    network.rip_learn(A_switch, C, f_max, stdp_weight=1, hebb_weight=1)
    network.rip_learn(Bz, C, f_max, stdp_weight=1, hebb_weight=1)
    network.rip_learn(Cz, D, f_max, stdp_weight=1, hebb_weight=1)
    # network.rip_learn(C_switch, A, f_max, stdp_weight=1, hebb_weight=1)
    network.rip_learn(Dz, A, f_max, stdp_weight=1, hebb_weight=1)

B_hat = pytorch_STDP.cart_to_polar(network.forward(Az))
B_polar = pytorch_STDP.cart_to_polar(B)
for i in range(torch.numel(B_hat[0])):
    print('{:.2f} - {:.2f}'.format(B_polar[0][0, i].item(), B_hat[0][0, i].item()))

# def construct_graph(N):

