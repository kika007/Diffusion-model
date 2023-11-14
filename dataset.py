import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader


# epoch = one forward and backward pass of ALL training samples
# batch_size = number of training samples used in one forward/backward pass
# number of iterations = number of passes, each pass (forward+backward) using [batch_size] number of sampes
# e.g : 100 samples, batch_size=20 -> 100/20=5 iterations for 1 epoch


class AudioDataset(Dataset):
    def __init__(self, dataset_path):
        self.wav_files = [os.path.join(dataset_path, file) for file in os.listdir(dataset_path) if file.endswith(".wav")]

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        file_path = self.wav_files[idx]
        waveform, sample_rate = torchaudio.load(file_path)
        return waveform, sample_rate, idx
    

dataset_path = "examples/data_dir"
dataset = AudioDataset(dataset_path)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

first_data = dataset[0]
features, rate, labels = first_data
print(features,rate,labels)





