import torch
import torchaudio
import matplotlib.pyplot as plt
from function.plot_signal import plot_signal
from function.play_signal import play_signal
from function.get_noise import get_noise
import numpy as np
from Model_demo import Denoiser
import torch.nn as nn
import torch.optim as optim



#IMPORT SIGNAL

file_path = "signal_examples/zvuk1.wav" 
signal, sample_rate = torchaudio.load(file_path)

#-------------------------------------------------


#PLOT SIGNAL

#plot_signal(signal,sample_rate)

#---------------------------------------------------


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

spectrogram = torchaudio.transforms.Spectrogram()(signal)

noisy_spectrogram = torchaudio.transforms.Spectrogram()(signal_noise)


# Trénovanie autoencodéra pre úpravu spektrogramov
criterion = nn.MSELoss()
optimizer = optim.Adam(denoiser_model.parameters(), lr=0.001)



# Uchováva stratu počas trénovania
losses = []



num_epochs = 20

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



# Predikcia denoised spektrogramu pomocou natrénovaného autoencodéra
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


# Vykreslenie priebehu strát počas trénovania
plt.plot(losses)
plt.title('Training Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.show()

#------------------------------------------------------------------------










