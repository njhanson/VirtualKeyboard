import mne
import numpy as np


# Load preprocessed epochs from .fif file

# Replace with your actual file name
epochs = mne.read_epochs("preprocessed-epo.fif", preload=True)

print("Loaded epochs:")
print(epochs)

# data shape: (n_trials, n_channels, n_times)

data = epochs.get_data()
# Shape: (n_trials, n_channels, n_times)

times = epochs.times


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