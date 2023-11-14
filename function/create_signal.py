import torch
import torchaudio
import soundfile as sf
import numpy as np

#parametrs
sample_rate=44100
frequency=1500
duration=3



t = torch.linspace(0, duration, int(duration * sample_rate), dtype=torch.float32)
signal = torch.sin(2 * torch.pi * frequency * t)

create_signal = "created_signals/new_audio1.wav"

sf.write(create_signal, signal, sample_rate, subtype='PCM_16', format='WAV')

