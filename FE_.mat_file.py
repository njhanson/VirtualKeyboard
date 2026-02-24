from scipy.io import loadmat
import numpy as np

# load .mat file

mat = loadmat("epochs.mat")

data = mat["data"]        # shape: (trials, channels, time)
times = mat["times"].squeeze()

print("Original shape:", data.shape)

# window selection: 300-600 ms (0.3-0.6 s)

tmin = 0.3
tmax = 0.6

time_mask = (times >= tmin) & (times <= tmax)

tensor = data[:, :, time_mask]

print("\nShape after window selection:")
print(tensor.shape)

#normalize per trial
eps = 1e-8

tensor_min = tensor.min(axis=(1, 2), keepdims=True)
tensor_max = tensor.max(axis=(1, 2), keepdims=True)
tensor = (tensor - tensor_min) / (tensor_max - tensor_min + eps)

print("\nShape after normalization:")
print(tensor.shape)

#save output as .npy 

np.save("snn_tensor.npy", tensor)

print("\nSaved as snn_tensor.npy")
print("Final shape:", tensor.shape)