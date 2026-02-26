# Feature extraction from .mat file for BE4999 SNN project, I used this to run .mat file data with 
# basic feature extraction (variance, mean, Hjorth parameters) and save as .npy for SNN input. We can 
# adjust time window and features as needed! - Madalyn 2-26-2026

from scipy.io import loadmat
import numpy as np 
import eeglib # for feature extraction, install via pip if needed (pip install eeglib)

# Load .mat file -- Michael is adding to GitHub, but adjust filename as needed
mat = loadmat("S17_Preprocessed_Epoch.mat")

# Extract datasets
target = mat["filtered_target_epochs"]
nontarget = mat["filtered_nontarget_epochs"]
times = mat["time"].squeeze()  # in seconds

print("Times min:", times.min())
print("Times max:", times.max())
print("Times shape:", times.shape) 

# Fix MATLAB->Python dimensions: (trials, channels, time) -- MATLAB used (channels, time, trials)
target = np.transpose(target, (2, 0, 1))
nontarget = np.transpose(nontarget, (2, 0, 1))

# Combine trials
X = np.concatenate([target, nontarget], axis=0)
y = np.concatenate([np.ones(target.shape[0]), np.zeros(nontarget.shape[0])])

print("X shape:", X.shape)
print("y shape:", y.shape)

# Time window selection
tmin, tmax = 300, 600 # ms
time_mask = (times >= tmin) & (times <= tmax)
X_window = X[:, :, time_mask]
print("Shape after time window selection:", X_window.shape)

# Normalize each trial (0–1)
eps = 1e-8
tensor_min = X_window.min(axis=(1, 2), keepdims=True)
tensor_max = X_window.max(axis=(1, 2), keepdims=True)
X_norm = (X_window - tensor_min) / (tensor_max - tensor_min + eps)
print("Shape after normalization:", X_norm.shape)

# Feature extraction per trial
# Example using EEGLib basic features: variance, mean, Hjorth parameters
feature_list = []
for trial in X_norm:  # trial: (channels, time)
    trial_features = []
    for ch in trial:
        
        var = np.var(ch)
        mean = np.mean(ch)

        # Hjorth manually --  activity, mobility, complexity
        diff1 = np.diff(ch)
        diff2 = np.diff(diff1)

        activity = np.var(ch)
        mobility = np.sqrt(np.var(diff1) / (activity + 1e-8))
        complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-8)) / (mobility + 1e-8)

        hjorth = (activity, mobility, complexity)

        trial_features.extend([var, mean] + list(hjorth))
    feature_list.append(trial_features)

features_array = np.array(feature_list)
print("Features array shape:", features_array.shape)

# Save tensor and features -- currently to desktop, adjust path as needed 



np.save("snn_tensor.npy", X_norm)
np.save("snn_features.npy", features_array)
print("Saved snn_tensor.npy and snn_features.npy")

# (nTrials x nFeatures) reshape for SNN input
tensor_reshaped = X_norm.reshape(X_norm.shape[0], -1)
print("Reshaped tensor for SNN input:", tensor_reshaped.shape)

# I opened the files from desktop using a folder in terminal and this code:

#import numpy as np

#X = np.load("snn_features.npy") and #X = np.load("snn_tensor.npy") 
#print(X.shape)
#print(X)
