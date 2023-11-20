import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torchaudio
import matplotlib.pyplot as plt

from function.get_noise import get_noise
from dataset import AudioDataset
from model_demo import Denoiser



#IMPORT DATASET

dataset_path = "examples/data_dir"
dataset = AudioDataset(dataset_path)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

#--------------------------------------------------

#DENOISER

num_epochs = 2

denoiser_model = Denoiser()

# Trénovanie autoencodéra pre úpravu spektrogramov
criterion = nn.MSELoss()
optimizer = optim.Adam(denoiser_model.parameters(), lr=0.001)

# Uchováva stratu počas trénovania
losses = []

for epoch in range(num_epochs):
    for i, inputs in enumerate(data_loader):
        waveform,sample_rate = inputs

        #vytváranie signal_noise
        #----------------------------------------------------
        SNR = 50

        noise = get_noise(waveform,SNR)

        for j in range(1):
            if(j ==0):
                signal_noise = waveform + noise

        signal_noise = signal_noise + noise
        #-------------------------------------------------------
        
        spectrogram = torchaudio.transforms.Spectrogram(n_fft=512)(waveform)

        noisy_spectrogram = torchaudio.transforms.Spectrogram(n_fft=512)(signal_noise)

        # Predikcia a výpočet chyby
        output_spectrogram = denoiser_model(noisy_spectrogram)
        loss = criterion(output_spectrogram, spectrogram)

        # Spätná propagácia a aktualizácia váh
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item()}')

# Vykreslenie priebehu strát počas trénovania
plt.plot(losses)
plt.title('Training Loss Over Time')
plt.xlabel('Steps')
plt.ylabel('MSE Loss')
plt.show()

