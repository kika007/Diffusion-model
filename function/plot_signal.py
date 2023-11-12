import torch
import torchaudio
import matplotlib.pyplot as plt
#import numpy as np
import librosa
#from IPython.display import Audio

def plot_signal(SPEECH_WAVEFORM, SAMPLE_RATE):

    def plot_waveform(waveform, sr, title="Waveform", ax=None):
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sr

        if ax is None:
            _, ax = plt.subplots(num_channels, 1)
        ax.plot(time_axis, waveform[0], linewidth=1)
        ax.grid(True)
        ax.set_xlim([0, time_axis[-1]])
        ax.set_title(title)


    def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
        if ax is None:
            _, ax = plt.subplots(1, 1)
        if title is not None:
            ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")

    
    spectrogram = torchaudio.transforms.Spectrogram(n_fft=512)
    spec = spectrogram(SPEECH_WAVEFORM)
    fig, axs = plt.subplots(2, 1)
    plot_waveform(SPEECH_WAVEFORM, SAMPLE_RATE, title="waveform", ax=axs[0])
    plot_spectrogram(spec[0], title="spectrogram", ax=axs[1])
    fig.tight_layout()
    plt.show()




    
    
