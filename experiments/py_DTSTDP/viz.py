import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import time
import copy


phases = np.linspace(0, 1, 51)
phases = list(zip(range(phases.size), phases))
rates = np.linspace(0, 100, 51)
rates = list(zip(range(rates.size), rates))

folder = '/Users/alexbaranski/Desktop/fig_folder/figure/'
timestamp = '1614827176'
stdp_file = os.path.join(folder, '{}_STDP.pkl'.format(timestamp))
rip_file = os.path.join(folder, '{}_RIP.pkl'.format(timestamp))

with open(stdp_file, 'rb') as f:
    stdp_dw = pickle.load(f)
with open(rip_file, 'rb') as f:
    rip_dw = pickle.load(f)

max_val = np.max([np.max(np.abs(stdp_dw)), np.max(np.abs(rip_dw))])
img1 = stdp_dw[0, 0, :, :]
img2 = rip_dw[0, 0, :, :]
fig = plt.figure(figsize=(8,4))
ax1 = plt.subplot(1, 2, 1)
plt.imshow(img1, vmin=-max_val, vmax=max_val, origin='upper', cmap='PiYG', extent=[0, 10, 0, 1])
ax1.set_aspect(10)
ax1.set_title('STDP learning rule')
ax1.set_xlabel('Post-synaptic rate')
ax1.set_ylabel('Post-synaptic phase')

ax2 = plt.subplot(1, 2, 2)
plt.imshow(img2, vmin=-max_val, vmax=max_val, origin='upper', cmap='PiYG', extent=[0, 10, 0, 1])
ax2.set_aspect(10)
ax2.set_title('RIP learning rule')
ax2.set_xlabel('Post-synaptic rate')
ax2.set_ylabel('Post-synaptic phase')

plt.suptitle('Weight change')

plt.savefig(os.path.join(folder, 'weight_change.eps'), pad_inches=0)
# plt.show()

# print(np.max(dW_array))
# print(np.min(dW_array))
# max_val = np.max([np.abs(np.max(dW_array)), np.abs(np.min(dW_array))])
# print(max_val)
# img = dW_array[0, 0, :, :]
# img[:, range(0, 101, 2)] = -100
# print(img.shape)
# ax = plt.imshow(img, vmin=-max_val, vmax=max_val, origin='lower')
# plt.title('Phase 1 vs. Phase 2')
# plt.ylabel('Post-synaptic phase')
# plt.xlabel('Post-synaptic rate')
# ax = plt.imshow(dW_array[:, 4, 0:50, 31])
# ax = plt.imshow(dW_array[:, 4, 0:50, 31])
# ax = plt.imshow(dW_array[:, 4, 0:50, 31])

# for i in range(101):
#     ax.set_data(dW_array[50, :, i, :])
#     plt.axis('off')
#     plt.title(str(i))
#     fig.canvas.draw()
#     time.sleep(0.01)
#     plt.draw()
#     plt.pause(0.01)
# plt.show()