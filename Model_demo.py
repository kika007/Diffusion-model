import torch.nn as nn
import torch

class Denoiser(nn.Module):
    def __init__(self):
        super(Denoiser, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # Adjusted to match the expected input channels
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1),   # Changed output channels to 1
        )

        # SNR processing layer
        self.snr_layer = nn.Sequential(
            nn.Linear(1, 64),  # Fully connected layer for SNR value
            nn.ReLU(inplace=True),
        )

    def forward(self, signal, time_step):
        # 'signal' represents the input signal, and 'snr_value' is the SNR value
        
        # Forward pass through the encoder
        x = self.encoder(signal)
        
        # Process SNR value through the SNR processing layer
        time_step = torch.tensor([time_step],dtype=torch.float)
        snr_processed = self.snr_layer(time_step)

        # Concatenate the encoded signal with the processed SNR value
        x = torch.cat((x, snr_processed.unsqueeze(-1).unsqueeze(-1).expand_as(x)))

        # Forward pass through the decoder
        x = self.decoder(x)
        return x






