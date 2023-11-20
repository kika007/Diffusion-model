import torch
import torchaudio
import matplotlib.pyplot as plt
from function.play_signal import play_signal
from function.get_noise import get_noise
from dataset import AudioDataset
import numpy as np
from model_demo import Denoiser
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader



#IMPORT SIGNAL

file_path = "signal_examples/zvuk1.wav" 
signal, sample_rate = torchaudio.load(file_path)

#-------------------------------------------------

#ADD NOISE

SNR = 50
noise = get_noise(signal,SNR)

for i in range(1):
    if(i ==0):
        signal_noise = signal + noise

    signal_noise = signal_noise + noise

#-------------------------------------------------------

#DENOISER

denoiser_model = Denoiser()

n_fft = 512

spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft)(signal)

noisy_spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft)(signal_noise)


# Trénovanie autoencodéra pre úpravu spektrogramov
criterion = nn.MSELoss()
optimizer = optim.Adam(denoiser_model.parameters(), lr=0.001)

# Uchováva stratu počas trénovania
losses = []

num_epochs = 10

for epoch in range(num_epochs):
    # Predikcia a výpočet chyby
    output_spectrogram = denoiser_model(noisy_spectrogram)
    loss = criterion(output_spectrogram, spectrogram)

    # Spätná propagácia a aktualizácia váh
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')



# Predikcia denoised spektrogramu pomocou natrénovaného denoisera
with torch.no_grad():
    denoised_spectrogram = denoiser_model(noisy_spectrogram)


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

#rekonštruovanie signálu - in progress problém že do modelu hádžem realne čísla a výstup su tiež realne čísla, takže to neviem transformovať späť

""""
reconstructed_signal = torch.istft(denoised_spectrogram,n_fft)

signal = signal.numpy()
plt.subplot(2,1,1)
plt.plot(signal[0])
plt.title('Original signal')

reconstructed_signal =reconstructed_signal.numpy()
plt.subplot(2,1,2)
plt.plot(reconstructed_signal[0])
plt.title('Reconstructed signal')

plt.show()
"""


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










