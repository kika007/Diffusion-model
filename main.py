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

n_steps = 200
SNR = 40
n_fft = 512

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

#denoiser_model = torch.load("trained_model/trained_model.pht")
#denoiser_model.eval()

# Trénovanie autoencodéra pre úpravu spektrogramov
criterion = nn.MSELoss()
optimizer = optim.Adam(denoiser_model.parameters(), lr=0.0001)

# Uchováva stratu počas trénovania
losses = []

num_epochs = 10

for epoch in range(num_epochs):

    time_step = torch.tensor([10])

    noisy_x = q_sample(signal, time_step, noise)
    # Predikcia a výpočet chyby

    spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft)(signal)

    noisy_spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft)(noisy_x)


    output_spectrogram = denoiser_model(noisy_spectrogram,time_step)
    loss = criterion(output_spectrogram, spectrogram)

    # Spätná propagácia a aktualizácia váh
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')



# Predikcia denoised spektrogramu pomocou natrénovaného denoisera
with torch.no_grad():
    denoised_spectrogram = denoiser_model(noisy_spectrogram,time_step)


# Vizualizácia pôvodného, zašumeného a denoised spektrogramu
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

#rekonštrukcia signálu s použitím funkcie torchaudio.transforms

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










