'''
This script is adapted from the utilities of the EINv2 code.
Credits to Yin Cao et al. Available from:
https://github.com/yinkalario/EIN-SELD
'''


import torch
import torch.nn as nn

from utils.stft import (STFT, LogmelFilterBank, intensityvector,
                                spectrogram_STFTInput)
import core.config as conf


class LogmelIntensity_Extractor(nn.Module):
    def __init__(self, conf):
        super().__init__()

        #data = cfg['data']
        sample_rate, n_fft, hop_length, window, n_mels, fmin, fmax = conf.input['fs'], conf.spectrog_param['n_fft'], \
        conf.spectrog_param['hoplen'], 'hann', conf.spectrog_param['numcep'], conf.spectrog_param['fmin'], \
        conf.spectrog_param['fmax']


        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        # STFT extractor
        self.stft_extractor = STFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
                                   window=window, center=center, pad_mode=pad_mode,
                                   freeze_parameters=True)

        # Spectrogram extractor
        self.spectrogram_extractor = spectrogram_STFTInput

        # Logmel extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=n_fft,
                                                 n_mels=n_mels, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db,
                                                 freeze_parameters=True)

        # Intensity vector extractor
        self.intensityVector_extractor = intensityvector

    def forward(self, x):
        """
        input:
            (batch_size, channels=4, data_length)
        output:
            (batch_size, channels, time_steps, freq_bins)
        """
        if x.ndim != 3:
            raise ValueError("x shape must be (batch_size, num_channels, data_length)\n \
                            Now it is {}".format(x.shape))
        x = self.stft_extractor(x)
        logmel = self.logmel_extractor(self.spectrogram_extractor(x))
        intensity_vector = self.intensityVector_extractor(x, self.logmel_extractor.melW)
        out = torch.cat((logmel, intensity_vector), dim=1)
        # time_step = (fs * wav_len / hop_size) + 1
        out = out[:,:,:-1,:]
        return out
