import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchaudio
import matplotlib.pyplot as plt

from function.get_noise import get_noise
from dataset import AudioDataset
from Model_demo import Denoiser

#SET PARAMETERS

n_steps = 20
SNR = 40
n_fft = 512
num_epochs = 2

#-----------------------------------------------

#IMPORT DATASET

dataset_path = "examples/data_dir"
dataset = AudioDataset(dataset_path)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

#--------------------------------------------------


#DEFINE TIMESTEP

alphas = 1. - torch.linspace(0.001, 0.2, n_steps)
alphas_cumprod = torch.cumprod(alphas, axis=0)

sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - sqrt_alphas_cumprod ** 2)

def q_sample(signal, t, noise=None):
    if noise is None:
        noise = get_noise(signal,SNR)
    return sqrt_alphas_cumprod.gather(-1, t) * signal + sqrt_one_minus_alphas_cumprod.gather(-1, t) * noise

#-------------------------------------------------

#DENOISER

denoiser_model = Denoiser()

criterion = nn.MSELoss()
optimizer = optim.Adam(denoiser_model.parameters(), lr=0.001)

losses = []

for epoch in range(num_epochs):
    for i, inputs in enumerate(data_loader):
        waveform,sample_rate = inputs

        
        time_step = torch.randint(0, n_steps, (1,)).item()
        time_step = torch.tensor([time_step])

        noise = get_noise(waveform,SNR)
        noisy_x = q_sample(waveform, time_step, noise)
        
        spectrogram = torchaudio.transforms.Spectrogram(n_fft=512)(waveform)

        noisy_spectrogram = torchaudio.transforms.Spectrogram(n_fft=512)(noisy_x)

        
        output_spectrogram = denoiser_model(noisy_spectrogram,time_step)
        loss = criterion(output_spectrogram, spectrogram)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item()}')

#-----------------------------------------------------------------------------

#SAVE MODEL

torch.save(denoiser_model,"trained_model/trained_model.pht")

#-----------------------------------------------------------------------------

# PLOT LOSS 
plt.plot(losses)
plt.title('Training Loss Over Time')
plt.xlabel('Steps')
plt.ylabel('MSE Loss')
plt.show()

#-----------------------------------------------------------------------------

