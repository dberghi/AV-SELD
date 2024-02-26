import torch
import torch.nn as nn
import torchaudio
import numpy as np
import os
import core.config as conf
from models.audio_encoder import AudioBackbone
from models.pytorch_i3d import InceptionI3d
from models.CMAF import CMAF_Layer
base_path = conf.input['project_path']

class AudioVisualConf(nn.Module):
    def __init__(self, device):
        super(AudioVisualConf, self).__init__()

        # visual front-end was precomputed and stored in h5py file
        if conf.training_param['visual_encoder_type'] == 'resnet':
            input_vis_dim = 4096
        elif conf.training_param['visual_encoder_type'] == 'i3d':
            input_vis_dim = 2048
        self.fc_visual = nn.Linear(input_vis_dim, 512) # reduce dim
        self.norm_visual = nn.LayerNorm(512) # good practice to have audio and visual embeddings normalized

        # audio front-end
        self.audioFrontEnd = AudioBackbone(device)
        self.audioFrontEnd.load_state_dict(
            torch.load(os.path.join(base_path, 'models', 'weights', 'audio_weights.pt'),
                       map_location=torch.device(device)))
        self.norm_audio = nn.LayerNorm(512) # good practice to have audio and visual embeddings normalized

        self.conformer = torchaudio.models.Conformer(input_dim=1024, num_heads=8, ffn_dim=1024, num_layers=4,
                                                    depthwise_conv_kernel_size=51, dropout=0.1)
        self.lengths = torch.from_numpy(
            np.ones(conf.training_param['batch_size']) * conf.training_param['num_video_frames']).to(device)

        # Audio-visual MLP back-end
        self.AVfnn_list = torch.nn.ModuleList()
        self.AVfnn_list.append(nn.Linear(1024, 128, bias=True))
        self.AVfnn_list.append(nn.Linear(128, 117, bias=True))


    def forward_visual(self, x):
        '''x -> (batch_size, time, embed_dim)'''
        x = self.fc_visual(x)  # (B, T, 512)
        return self.norm_visual(x)

    def forward_audio(self, x):
        '''x -> (batch_size, feature_maps, time_steps, mel_bins)'''
        x = self.audioFrontEnd(x) # (B, T, 512)
        return self.norm_audio(x)


    def forward(self, audio_feat, visual_feat):
        audio_embedding = self.forward_audio(audio_feat)
        visual_embedding = self.forward_visual(visual_feat)
        # concat embeddings for AV-conformer
        x = torch.cat((audio_embedding, visual_embedding), 2)

        if len(self.lengths) == len(x):
            lengths = self.lengths
        else:
            lengths = self.lengths[:len(x)]
        x, _ = self.conformer(x, lengths)

        for fnn_cnt in range(len(self.AVfnn_list) - 1):
            x = self.AVfnn_list[fnn_cnt](x)
        doa = torch.tanh(self.AVfnn_list[-1](x)) # MAIN OUTPUT

        return doa


class AudioVisualCMAF(nn.Module):
    def __init__(self, device):
        super(AudioVisualCMAF, self).__init__()

        # visual front-end was precomputed and stored in h5py file
        if conf.training_param['visual_encoder_type'] == 'resnet':
            input_vis_dim = 4096
        elif conf.training_param['visual_encoder_type'] == 'i3d':
            input_vis_dim = 2048
        self.fc_visual = nn.Linear(input_vis_dim, 512) # reduce dim
        self.norm_visual = nn.LayerNorm(512) # good practice to have audio and visual embeddings normalized

        # audio front-end
        self.audioFrontEnd = AudioBackbone(device)
        self.audioFrontEnd.load_state_dict(
            torch.load(os.path.join(base_path, 'models', 'weights', 'audio_weights.pt'),
                       map_location=torch.device(device)))
        self.norm_audio = nn.LayerNorm(512) # good practice to have audio and visual embeddings normalized

        # cross-attention
        self.crossA2V_list = torch.nn.ModuleList()
        self.crossV2A_list = torch.nn.ModuleList()
        for cmaf_count in range(conf.training_param['num_cmaf_layers']):  # original CMAF uses 4 heads (but 6 layers)
            self.crossA2V_list.append(CMAF_Layer(d_model=512, nhead=8))
            self.crossV2A_list.append(CMAF_Layer(d_model=512, nhead=8))


        # Audio-visual MLP back-end
        self.AVfnn_list = torch.nn.ModuleList()
        self.AVfnn_list.append(nn.Linear(1024, 128, bias=True))
        self.AVfnn_list.append(nn.Linear(128, 117, bias=True))


    def forward_visual(self, x):
        '''x -> (batch_size, time, embed_dim)'''
        x = self.fc_visual(x)  # (B, T, 512)
        return self.norm_visual(x)

    def forward_audio(self, x):
        '''x -> (batch_size, feature_maps, time_steps, mel_bins)'''
        x = self.audioFrontEnd(x) # (B, T, 512)
        return self.norm_audio(x)

    def forward_cross_attention(self, x1, x2):
        for cmaf_cnt in range(len(self.crossA2V_list)):
            x1 = self.crossA2V_list[cmaf_cnt](alpha=x1, beta=x2)
            x2 = self.crossV2A_list[cmaf_cnt](alpha=x2, beta=x1)
        return x1, x2


    def forward(self, audio_feat, visual_feat):
        audio_embedding = self.forward_audio(audio_feat)
        visual_embedding = self.forward_visual(visual_feat)
        # forward CMAF cross-attention
        A2V_emb, V2A_emb = self.forward_cross_attention(audio_embedding, visual_embedding)
        # concat
        x = torch.cat((A2V_emb, V2A_emb), 2)

        for fnn_cnt in range(len(self.AVfnn_list) - 1):
            x = self.AVfnn_list[fnn_cnt](x)
        doa = torch.tanh(self.AVfnn_list[-1](x)) # MAIN OUTPUT

        return doa




class MSELoss_ADPIT(object):
    '''
    This class is taken from the DCASE Task 3 baseline code.
    Credits to Adavanne et al. Available from:
    https://github.com/sharathadavanne/seld-dcase2023
    '''
    def __init__(self):
        super().__init__()
        self._each_loss = nn.MSELoss(reduction='none')

    def _each_calc(self, output, target):
        return self._each_loss(output, target).mean(dim=(2))  # class-wise frame-level

    def __call__(self, output, target):
        """
        Auxiliary Duplicating Permutation Invariant Training (ADPIT) for 13 (=1+6+6) possible combinations
        Args:
            output: [batch_size, frames, num_track*num_axis*num_class=3*3*12]
            target: [batch_size, frames, num_track_dummy=6, num_axis=4, num_class=12]
        Return:
            loss: scalar
        """
        target_A0 = target[:, :, 0, 0:1, :] * target[:, :, 0, 1:, :]  # A0, no ov from the same class, [batch_size, frames, num_axis(act)=1, num_class=12] * [batch_size, frames, num_axis(XYZ)=3, num_class=12]
        target_B0 = target[:, :, 1, 0:1, :] * target[:, :, 1, 1:, :]  # B0, ov with 2 sources from the same class
        target_B1 = target[:, :, 2, 0:1, :] * target[:, :, 2, 1:, :]  # B1
        target_C0 = target[:, :, 3, 0:1, :] * target[:, :, 3, 1:, :]  # C0, ov with 3 sources from the same class
        target_C1 = target[:, :, 4, 0:1, :] * target[:, :, 4, 1:, :]  # C1
        target_C2 = target[:, :, 5, 0:1, :] * target[:, :, 5, 1:, :]  # C2

        target_A0A0A0 = torch.cat((target_A0, target_A0, target_A0), 2)  # 1 permutation of A (no ov from the same class), [batch_size, frames, num_track*num_axis=3*3, num_class=12]
        target_B0B0B1 = torch.cat((target_B0, target_B0, target_B1), 2)  # 6 permutations of B (ov with 2 sources from the same class)
        target_B0B1B0 = torch.cat((target_B0, target_B1, target_B0), 2)
        target_B0B1B1 = torch.cat((target_B0, target_B1, target_B1), 2)
        target_B1B0B0 = torch.cat((target_B1, target_B0, target_B0), 2)
        target_B1B0B1 = torch.cat((target_B1, target_B0, target_B1), 2)
        target_B1B1B0 = torch.cat((target_B1, target_B1, target_B0), 2)
        target_C0C1C2 = torch.cat((target_C0, target_C1, target_C2), 2)  # 6 permutations of C (ov with 3 sources from the same class)
        target_C0C2C1 = torch.cat((target_C0, target_C2, target_C1), 2)
        target_C1C0C2 = torch.cat((target_C1, target_C0, target_C2), 2)
        target_C1C2C0 = torch.cat((target_C1, target_C2, target_C0), 2)
        target_C2C0C1 = torch.cat((target_C2, target_C0, target_C1), 2)
        target_C2C1C0 = torch.cat((target_C2, target_C1, target_C0), 2)

        output = output.reshape(output.shape[0], output.shape[1], target_A0A0A0.shape[2], target_A0A0A0.shape[3])  # output is set the same shape of target, [batch_size, frames, num_track*num_axis=3*3, num_class=12]
        pad4A = target_B0B0B1 + target_C0C1C2
        pad4B = target_A0A0A0 + target_C0C1C2
        pad4C = target_A0A0A0 + target_B0B0B1
        loss_0 = self._each_calc(output, target_A0A0A0 + pad4A)  # padded with target_B0B0B1 and target_C0C1C2 in order to avoid to set zero as target
        loss_1 = self._each_calc(output, target_B0B0B1 + pad4B)  # padded with target_A0A0A0 and target_C0C1C2
        loss_2 = self._each_calc(output, target_B0B1B0 + pad4B)
        loss_3 = self._each_calc(output, target_B0B1B1 + pad4B)
        loss_4 = self._each_calc(output, target_B1B0B0 + pad4B)
        loss_5 = self._each_calc(output, target_B1B0B1 + pad4B)
        loss_6 = self._each_calc(output, target_B1B1B0 + pad4B)
        loss_7 = self._each_calc(output, target_C0C1C2 + pad4C)  # padded with target_A0A0A0 and target_B0B0B1
        loss_8 = self._each_calc(output, target_C0C2C1 + pad4C)
        loss_9 = self._each_calc(output, target_C1C0C2 + pad4C)
        loss_10 = self._each_calc(output, target_C1C2C0 + pad4C)
        loss_11 = self._each_calc(output, target_C2C0C1 + pad4C)
        loss_12 = self._each_calc(output, target_C2C1C0 + pad4C)

        loss_min = torch.min(
            torch.stack((loss_0,
                         loss_1,
                         loss_2,
                         loss_3,
                         loss_4,
                         loss_5,
                         loss_6,
                         loss_7,
                         loss_8,
                         loss_9,
                         loss_10,
                         loss_11,
                         loss_12), dim=0),
            dim=0).indices

        loss = (loss_0 * (loss_min == 0) +
                loss_1 * (loss_min == 1) +
                loss_2 * (loss_min == 2) +
                loss_3 * (loss_min == 3) +
                loss_4 * (loss_min == 4) +
                loss_5 * (loss_min == 5) +
                loss_6 * (loss_min == 6) +
                loss_7 * (loss_min == 7) +
                loss_8 * (loss_min == 8) +
                loss_9 * (loss_min == 9) +
                loss_10 * (loss_min == 10) +
                loss_11 * (loss_min == 11) +
                loss_12 * (loss_min == 12)).mean()

        return loss
