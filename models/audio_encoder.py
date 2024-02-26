#!/usr/bin/python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio #.models.conformer as conformer
import core.config as conf

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):

        super(ConvBlock, self).__init__()

        self.downsample_res1x1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(1,1), stride=(2,2),
                               bias=False)

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)

        self.bn_res = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_type='avg', pool_size=(2, 2)):
        residual = self.bn_res(self.downsample_res1x1(x))

        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'frac':
            fractional_maxpool2d = nn.FractionalMaxPool2d(kernel_size=pool_size, output_ratio=1 / np.sqrt(2))
            x = fractional_maxpool2d(x)

        x += residual # residual loop
        return x


class AudioBackbone(nn.Module):
    def __init__(self, device, pool_type='avg'):
        super(AudioBackbone, self).__init__()

        self.pool_type = pool_type

        self.conv_block1 = ConvBlock(in_channels=7, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.conformer = torchaudio.models.Conformer(input_dim=512, num_heads=8, ffn_dim=1024, num_layers=4,
                                                     depthwise_conv_kernel_size=51, dropout=0.1)

        self.lengths = torch.from_numpy(
            np.ones(conf.training_param['batch_size']) * conf.training_param['num_video_frames']).to(device)


    def forward(self, x): #, one_hot_tensor):
        '''input: (batch_size, channels, time_steps, mel_bins) e.g.(64, 16, 960, 64)'''

        x = self.conv_block1(x, self.pool_type)
        x = self.conv_block2(x, self.pool_type)
        x = self.conv_block3(x, self.pool_type)
        x = self.conv_block4(x, self.pool_type)
        '''(batch_size, feature_maps, time_steps, mel_bins)'''

        if self.pool_type == 'avg':
            x = torch.mean(x, dim=3)
        elif self.pool_type == 'max':
            (x, _) = torch.max(x, dim=3)
        '''(batch_size, feature_maps, time_steps)'''

        audio_features = x.transpose(1, 2)
        ''' (batch_size, time_steps, feature_maps):'''

        if len(self.lengths) == len(x):
            lengths = self.lengths
        else:
            lengths = self.lengths[:len(x)]
        audio_features, _ = self.conformer(audio_features, lengths)

        return audio_features #, audio_predictions