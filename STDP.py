import numpy as np
import itertools
from multiprocessing import Pool
import os
import pickle
import random
import time


def get_time_stamp(seconds):
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    seconds = int(seconds)
    return '{} : {} : {}'.format(int(hours), str(int(minutes)).rjust(2, '0'), str(int(seconds)).rjust(2, '0'))


def STDP_1(pre_t, post_t, params):
    x = post_t - pre_t
    A_pos = params['A_pos']
    A_neg = params['A_neg']
    t_pos = params['t_pos']
    t_neg = params['t_neg']

    if x == 0:
        return 0
    elif x > 0:
        return A_pos * np.exp(-x / t_pos)
    else:
        return -A_neg * np.exp(x / t_neg)


def STDP_2(pre_t, post_t, params):
    x = post_t - pre_t

    if x == 0:
        return 0
    elif x > 0:
        return -x + 1
    else:
        return -x - 1


def sum_spikes(spikes_1, spikes_2, func, params):
    return sum([func(pre_t, post_t, params) for pre_t, post_t in itertools.product(spikes_1, spikes_2)])


def get_spikes(phase, rate):
    if rate == 0:
        return np.empty(0)
    else:
        return np.mod(np.arange(0, 1, 1/rate) + phase, 1)


def jitter_spikes(spike_train, noise_level):
    spike_train = spike_train + (np.random.rand(spike_train.size)-1/2)*2*noise_level
    # spike_train = np.array([i for i in spike_train if 0 <= i < 1])
    return spike_train

def get_dW(nxn):
    params = {
        'A_pos': 1,
        'A_neg': 1,
        't_pos': 1/4,
        't_neg': 1/4
    }

    p0 = nxn[0][0]
    r0 = nxn[0][1]
    p1 = nxn[1][0]
    r1 = nxn[1][1]

    index = (p0[0], r0[0], p1[0], r1[0])
    spikes_1 = get_spikes(p0[1], r0[1])
    spikes_2 = get_spikes(p1[1], r1[1])
    dW = sum_spikes(spikes_1, spikes_2, STDP_2, params)
    # dW = np.mean([sum_spikes(jitter_spikes(spikes_1, 0.01), jitter_spikes(spikes_2, 0.01), STDP_1, params) for i in range(5)])

    return (index, dW)

def get_dW_RIP(nxn):
    p0 = nxn[0][0]
    r0 = nxn[0][1]
    p1 = nxn[1][0]
    r1 = nxn[1][1]

    stdp_weight = 1
    hebb_weight = 1/16

    index = (p0[0], r0[0], p1[0], r1[0])
    a = 4
    hebb_term = r0[1] * r1[1]
    phase_diff = p1[1] - p0[1]
    stdp_term = 2 * np.floor(phase_diff / 2) + 1 - phase_diff

    # print(hebb_term, stdp_term)
    factor = 8 * a * hebb_term / ((a + 2 * hebb_term) ** 2)
    # factor = 1
    dW = factor * stdp_weight * stdp_term + hebb_weight * hebb_term

    return (index, dW)


def get_chunks(lst, n):
    chunks = list()
    for i in range(0, len(lst), n):
        chunks.append(lst[i:i+n])
    return chunks


def gen_range(*args):
    if len(args) == 1 and type(args[0]) == tuple:
        arg = args[0]
        values = np.linspace(*arg)
    else:
        values = np.array(args)
    values = list(zip(range(values.size), values))
    return values


def gen_STDP_diagram(pre_phases, pre_rates, post_phases, post_rates, func, folder, filename, time_stamp):
    print('Initializing phase and rate arrays...')
    pre_phase_x_rate = list(itertools.product(pre_phases, pre_rates))
    post_phase_x_rate = list(itertools.product(post_phases, post_rates))
    neuron_x_neuron = list(itertools.product(pre_phase_x_rate, post_phase_x_rate))
    print('Shuffling data...')
    random.shuffle(neuron_x_neuron)
    pr_x_pr = np.zeros((len(pre_phases), len(pre_rates), len(post_phases), len(post_rates)))
    print('beginning...')

    N = len(neuron_x_neuron)
    start_time = time.time()
    check_time = time.time()
    with Pool(35) as p:
        res = p.imap(func, neuron_x_neuron)
        for j in range(N):
            x = res.next()
            pr_x_pr[x[0]] = x[1]
            t = time.time()
            if t - check_time > 10:
                run_time = t - start_time
                T = N / (j + 1) * run_time
                ETA = T - run_time
                print('ETA: {}'.format(get_time_stamp(ETA)))
                check_time = t

    print('Writing file...')
    filepath = os.path.join(folder, '{}_{}.pkl'.format(time_stamp, filename))
    with open(filepath, 'wb') as f:
        pickle.dump(pr_x_pr, f)


if __name__ == '__main__':
    phases_pre = gen_range(.5)
    rates_pre = gen_range(1)
    phases_post = gen_range((0, 1, 501))
    rates_post = gen_range((0, 10, 501))

    folder = '/Users/alexbaranski/Desktop/fig_folder/figure/'
    time_stamp = str(int(time.time()))

    print(time_stamp)

    gen_STDP_diagram(phases_pre, rates_pre, phases_post, rates_post, get_dW, folder, 'STDP', time_stamp)
    gen_STDP_diagram(phases_pre, rates_pre, phases_post, rates_post, get_dW_RIP, folder, 'RIP', time_stamp)



    # print('Initializing phase and rate arrays...')
    # pre_phase_x_rate = list(itertools.product(pre_phases, pre_rates))
    # post_phase_x_rate = list(itertools.product(post_phases, post_rates))
    # neuron_x_neuron = list(itertools.product(pre_phase_x_rate, post_phase_x_rate))
    # print('Shuffling data...')
    # random.shuffle(neuron_x_neuron)
    # print('Dividing data into chunks...')
    # # data_chunks = get_chunks(neuron_x_neuron, 10000)
    # pr_x_pr = np.zeros((len(pre_phases), len(pre_rates), len(post_phases), len(post_rates)))
    # print('beginning...')
    #
    # N = len(neuron_x_neuron)
    # start_time = time.time()
    # check_time = time.time()
    # with Pool(35) as p:
    #     res = p.imap(get_dW, neuron_x_neuron)
    #     for j in range(N):
    #         x = res.next()
    #         pr_x_pr[x[0]] = x[1]
    #         t = time.time()
    #         if t - check_time > 10:
    #             run_time = t - start_time
    #             T = N / (j + 1) * run_time
    #             ETA = T - run_time
    #             # print(ETA)
    #             print('ETA: {}'.format(get_time_stamp(ETA)))
    #             check_time = t
    #
    # print('Writing file...')
    # folder = '/Users/alexbaranski/Desktop/stdp_data'
    # time_stamp = str(int(time.time()))
    # filepath = os.path.join(folder, '{}_stdp.pkl'.format(time_stamp))
    # with open(filepath, 'wb') as f:
    #     pickle.dump(pr_x_pr, f)

