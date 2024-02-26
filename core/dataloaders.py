#!/usr/bin/python
import os.path

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset

import core.config as conf
import utils.utils as utils



data_path = conf.input['feature_path']


class load_data_from_scratch(Dataset):
    def __init__(self, dataset_stats, sequences_paths_list, visual_encoder, train_or_test='train'):
        self.sequences_paths_list = sequences_paths_list
        self.visual_encoder = visual_encoder
        self.train_or_test = train_or_test
        self.enumerated_chunks = self.enumerate_chunks(dataset_stats, conf.input[
            'input_step_train'] if self.train_or_test == 'train' else conf.input[
            'input_step_test'])  # list of [sequence, label_start, label_end]

    def __len__(self):
        if self.train_or_test == 'train':
            return int(len(self.enumerated_chunks) * 8) # 8 ACS transformations
        else:
            return int(len(self.enumerated_chunks))

    def __getitem__(self, chunk_idx):
        if self.train_or_test == 'train':
            ACS_case = int(chunk_idx / len(self.enumerated_chunks)) + 1 # not elegant yet effective...
        else:
            ACS_case = 3 # φ = φ, θ = θ
        #ACS_case = 3 # REMOVE FOR CONDOR!!!!!!!
        idx = chunk_idx % len(self.enumerated_chunks)
        sequence = self.enumerated_chunks[idx][0]
        audio_sequence = '{}.wav'.format(sequence)

        # find path corresponding to audio_sequence
        for tmp_path in self.sequences_paths_list:
            if os.path.basename(tmp_path) == audio_sequence:
                audio_path = os.path.join(conf.input['data_path'], tmp_path)

        # load audio files
        start_time = self.enumerated_chunks[idx][1] / conf.input['label_resolution']
        audio, fs = utils.load_audio_file(audio_path, start_time=start_time)
        audio = utils.swap_audio_channels(audio, ACS_case)
        audio = np.swapaxes(audio, 0, 1)

        # load labels
        label_path = os.path.join(conf.input['feature_path'], self.train_or_test, str(ACS_case),
                                  '{}.npy'.format(sequence))
        full_label = np.load(label_path)
        # pad label if needed
        if self.enumerated_chunks[idx][2] > full_label.shape[0]:
            zero_pad = np.zeros((self.enumerated_chunks[idx][2] - full_label.shape[0], 6, 4, conf.input['num_classes']))
            full_label = np.concatenate((full_label, zero_pad), axis=0)
        #if os.path.basename(label_path) == 'fold3_room12_mix007.npy':

        target_label = full_label[self.enumerated_chunks[idx][1]:self.enumerated_chunks[idx][2]]

        # load video frames
        # find path corresponding to audio_sequence
        frames_path = audio_path.replace('foa', 'frames').replace('mic', 'frames').replace('wav', 'mp4')
        frames_path = '{}{}/{}'.format(frames_path.split('fold')[0], str(ACS_case), os.path.basename(frames_path))
        if self.visual_encoder == 'resnet':
            frames = utils.load_rgb_frames_PIL(frames_path, start_time)
            frames = torch.stack(frames)
        elif self.visual_encoder == 'i3d':
            frames_I3D = utils.load_rgb_frames(frames_path, start_time)
            frames_left = frames_I3D[:, :, 0:224, :]
            frames_right = frames_I3D[:, :, -224:, :]
            frames_left = np.moveaxis(frames_left, 3, 0)  # (c,t,h,w)
            frames_right = np.moveaxis(frames_right, 3, 0)  # (c,t,h,w)
            frames_left = torch.from_numpy(frames_left)
            frames_right = torch.from_numpy(frames_right)
            frames = [frames_left, frames_right]
        else:
            raise ValueError(
                """Input feature '%s' non supported. Use 'resnet' or 'i3d' instead.""" % self.visual_encoder)

        audio = torch.from_numpy(audio)
        target_label = torch.from_numpy(target_label)

        #return audio, frames, frames_left, frames_right, target_label, sequence, start_time  # start_time is useful to load video frames later
        return audio, frames, target_label, sequence, start_time  # start_time is useful to load video frames later

    def enumerate_chunks(self, dataset_stats, input_step_sec):
        enumerated_chunks = []
        for sequence in dataset_stats:
            num_labels = dataset_stats[sequence][0]
            hop_labels = int(conf.input['label_resolution'] * input_step_sec)
            tot_steps = int(np.ceil(num_labels / hop_labels))
            for label_step in range(tot_steps):
                lab_idx_start = int(label_step * hop_labels)
                lab_idx_end = int(lab_idx_start + (conf.input['label_resolution'] * conf.input['input_len_sec']))
                enumerated_chunks.append([sequence, lab_idx_start, lab_idx_end])
        return enumerated_chunks




class load_data_from_hdf5(Dataset):
    def __init__(self, h5py_dir, normalize=False, mean=None, std=None):
        self.h5_file = h5py.File(h5py_dir, 'r')
        self.normalize = normalize
        self.mean = mean
        self.std = std

    def __len__(self):
        return int((self.h5_file['audio_feat'].shape[0]) / 1)

    def __getitem__(self, chunk_idx):

        audio_features = self.h5_file['audio_feat'][chunk_idx]
        visual_features = self.h5_file['video_feat'][chunk_idx]
        labels = self.h5_file['labels'][chunk_idx]
        sequence = self.h5_file['sequences'][chunk_idx].decode("utf-8")
        initial_time = self.h5_file['initial_time'][chunk_idx]

        if self.normalize:
            # Normalize feature
            n_scaler_chan = self.mean.shape[0]
            audio_features = (audio_features - self.mean) / self.std

        audio_features = torch.from_numpy(audio_features)
        visual_features = torch.from_numpy(visual_features)
        labels = torch.from_numpy(labels).float()

        return audio_features, visual_features, labels, initial_time, sequence
        #return audio_features, labels, initial_time, sequence