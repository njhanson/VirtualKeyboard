import scipy.io
import numpy as np
from scipy.signal import butter, filtfilt

# Load the .mat file into Python workspace
data = scipy.io.loadmat(r'C:\Users\MikeK\Downloads\s17.mat', squeeze_me=True, struct_as_record=False)
t2 = data['train'][0]

# lets us see where target stimulus and where non target stim is
target_indices = np.where(t2.markers_target == 1)[0]
nontarget_indices = np.where(t2.markers_target == 2)[0]

# define parameters, so sampling rate (512hz) and epoch length (600ms)
samplingrate = t2.srate
epoch_length = round(0.6 * samplingrate)

# Design bandpass filter (0.5–15 Hz)
b, a = butter(4, [0.5 / (samplingrate / 2), 15 / (samplingrate / 2)], btype='bandpass')

# Use raw EEG data
raw_data = t2.data

pz_index = 13  # example electrode, we can choose many but a paper mentions Pz/13 is strongest

time = np.arange(epoch_length) / samplingrate * 1000

# lets us create these arrays, channels, samples, and number of trials we
target_epochs = np.zeros((32, epoch_length, len(target_indices)))
nontarget_epochs = np.zeros((32, epoch_length, len(nontarget_indices)))

# Extracts all target trials
for i, firstargetstimulus in enumerate(target_indices):
    target_epochs[:, :, i] = raw_data[:, firstargetstimulus:firstargetstimulus + epoch_length]

# extracts all nontarget trials
for i, firstargetstimulus in enumerate(nontarget_indices):
    nontarget_epochs[:, :, i] = raw_data[:, firstargetstimulus:firstargetstimulus + epoch_length]

filtered_target_epochs = np.zeros_like(target_epochs)
filtered_nontarget_epochs = np.zeros_like(nontarget_epochs)

for i in range(len(target_indices)):
    filtered_target_epochs[:, :, i] = filtfilt(b, a, target_epochs[:, :, i].T).T

for i in range(len(nontarget_indices)):
    filtered_nontarget_epochs[:, :, i] = filtfilt(b, a, nontarget_epochs[:, :, i].T).T

# Average across trials
average_target = np.mean(filtered_target_epochs, axis=2)
average_nontarget = np.mean(filtered_nontarget_epochs, axis=2)

# Compiling the dataset
output_file = r'C:\Users\MikeK\Downloads\DatasetMatfiles\S17_Preprocessed_Epoch.mat'

numberof_target = filtered_target_epochs.shape[2]
numberof_nontarget = filtered_nontarget_epochs.shape[2]

# when calling, 1's are filtered targets and 0's are filtered nontargets
labelsfortargetandnontarget = np.concatenate((np.ones(numberof_target), np.zeros(numberof_nontarget)))

all_epochs = np.concatenate((filtered_target_epochs, filtered_nontarget_epochs), axis=2)

scipy.io.savemat(output_file, {
    'filtered_target_epochs': filtered_target_epochs,
    'filtered_nontarget_epochs': filtered_nontarget_epochs,
    'all_epochs': all_epochs,
    'labelsfortargetandnontarget': labelsfortargetandnontarget,
    'average_target': average_target,
    'average_nontarget': average_nontarget,
    'samplingrate': samplingrate,
    'time': time,
    'pz_index': pz_index
})

print('Preprocessed EEG epochs saved successfully to Downloads folder.')