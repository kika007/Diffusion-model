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
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1),   
        )

        # Time_step processing layer
        self.time_layer = nn.Sequential(
            nn.Linear(1, 64),  
            nn.ReLU(inplace=True),
        )

    def forward(self, signal, time_step):
        
        # Forward pass through the encoder
        x = self.encoder(signal)
        
        time_step = torch.tensor([time_step],dtype=torch.float)
        snr_processed = self.time_layer(time_step)

        # Concatenate the encoded signal with the processed time_step value
        x = torch.cat((x, snr_processed.unsqueeze(-1).unsqueeze(-1).expand_as(x)))

        # Forward pass through the decoder
        x = self.decoder(x)
        return x






