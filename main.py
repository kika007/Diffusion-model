import torch
import torchaudio
import matplotlib.pyplot as plt
from function.play_signal import play_signal
from function.get_noise import get_noise
from dataset import AudioDataset
import numpy as np
from Model_demo import Denoiser
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

#SET PARAMETERS

n_steps = 20
SNR = 40
n_fft = 512
num_epochs = 1

#-----------------------------------------------

#IMPORT SIGNAL

file_path = "examples\data_dir\MIDI-Unprocessed_01_R1_2006_01-09_ORIG_MID--AUDIO_01_R1_2006_02_Track02_wav.wav" 
signal, sample_rate = torchaudio.load(file_path)

#-------------------------------------------------

#ADD NOISE

noise = get_noise(signal,SNR)

#-------------------------------------------------------

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

#load trained model
#denoiser_model = torch.load("trained_model/trained_model.pht")
#denoiser_model.eval()

#loss and optimalization
criterion = nn.MSELoss()
optimizer = optim.Adam(denoiser_model.parameters(), lr=0.0001)

#losses storage
losses = []


for epoch in range(num_epochs):

    #Setting random time step
    time_step = torch.randint(0, n_steps, (1,)).item()
    time_step = torch.tensor([time_step])

    #getting noisy signal
    noisy_x = q_sample(signal, time_step, noise)


    spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft)(signal)
    noisy_spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft)(noisy_x)


    output_spectrogram = denoiser_model(noisy_spectrogram,time_step)
    loss = criterion(output_spectrogram, spectrogram)

    # Backpropagation and optimalization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')



# Pprediction of denoiser
with torch.no_grad():
    denoised_spectrogram = denoiser_model(noisy_spectrogram,time_step)


#-----------------------------------------------------------------------


#SPECTROGRAMS

plt.figure(figsize=(18, 12))

plt.subplot(3, 1, 1)
plt.imshow(torch.log(spectrogram[0] + 1e-9).numpy(), aspect='auto', origin='lower')
plt.title('Original Spectrogram')

plt.subplot(3, 1, 2)
plt.imshow(torch.log(noisy_spectrogram[0] + 1e-9).numpy(), aspect='auto', origin='lower')
plt.title('Noisy Spectrogram')

plt.subplot(3, 1, 3)
plt.imshow(torch.log(denoised_spectrogram[0] + 1e-9).numpy(), aspect='auto', origin='lower')
plt.title('Denoised Spectrogram')

plt.show()

#------------------------------------------------------------------------

#RECONSTRACTION

reconstructed_signal = torchaudio.transforms.GriffinLim(n_fft=n_fft)(denoised_spectrogram)
signal_1 = signal.numpy()
plt.subplot(2,1,1)
plt.plot(signal_1[0])
plt.title('Original signal')

signal_new_1 = reconstructed_signal.numpy()
plt.subplot(2,1,2)
plt.plot(signal_new_1[0])
plt.title('transformed signal')

plt.show()

#------------------------------------------------------------------------









