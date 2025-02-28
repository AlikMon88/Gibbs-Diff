import os
import numpy as np
import matplotlib.pyplot as plt
import random

def create_1d_data(n_depth = 100, n_samples=1000, decay=0.1):
    
    signal_arr, noise_arr, obs_arr = [], [], []

    for _ in range(n_samples):

        x_vals = np.linspace(-5, 5, n_depth)
        signal = np.sin(x_vals)
        noise = np.random.normal(0, 1, size=n_depth)

        _r = np.random.rand(n_depth)
        observation = _r * signal + (1 - _r) * decay * noise

        signal_arr.append(signal)
        noise_arr.append(noise)
        obs_arr.append(observation)

    return np.array(obs_arr), np.array(signal_arr), np.array(noise_arr) 

if __name__ == '__main__':
    
    print('Running ... __noise_create.py__ ...')

    n_samples = 10000
    rn = random.randint(0, n_samples)

    observation, signal, noise = create_1d_data(n_samples=n_samples)
    print(observation.shape, signal.shape, noise.shape)

    plt.figure(figsize=(10, 5))

    plt.plot(observation[rn], label="Observation", alpha=0.8, marker='.')
    plt.plot(signal[rn], label="Signal", linestyle="dashed", alpha=0.7, color = 'red')
    plt.legend()
    plt.show()
