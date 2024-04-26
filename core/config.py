#!/usr/bin/python

import torch, os



input = {
    'project_path': '/mnt/fast/nobackup/users/db00785/AV-SELD/', #,os.getcwd()
    'data_path': '/mnt/fast/nobackup/scratch4weeks/db00785/data_dcase2023_task3/', # path to dataset
    'feature_path': '/mnt/fast/nobackup/scratch4weeks/db00785/data_dcase2023_task3/features/', # might want to save h5py (and scaler) somewhere else.
    'features': 'IV',
    'fps': 10,
    'fs': 24000, # Hz
    'input_len_sec': 3, # seconds
    'input_step_train': 0.5, #seconds
    'input_step_test': 3, #seconds (no overlap)
    'num_classes': 13,
    'audio_format': 'foa', # 'foa' or 'mic'
    'label_resolution': 10,
}

training_param = {
    'optimizer': torch.optim.Adam,
    #'criterion': nn.CrossEntropyLoss,
    'learning_rate': 0.00005, # default if user does not provide a different lr to the parser
    'epochs': 10, # default if user does not provide a different number to the parser
    'batch_size': 32, # default if user does not provide a different size to the parser
    'frame_len_samples': input['input_len_sec'] * input['fs'], # number of audio samples in input_len_sec,
    'num_video_frames': input['input_len_sec'] * input['fps'], # number of video frames in input_len_sec,
    'visual_encoder_type': 'resnet', # choose between 'resnet' or 'i3d'
    'model_type': 'conformer', # choose between 'conformer' or 'cmaf'
    'num_heads': 8, # number of heads in MHSA-MHCA
    'num_cmaf_layers': 4,
}

spectrog_param = { # used for log mel spec, gcc-phat, or salsa, or IV
    'winlen': 512, # samples
    'hoplen': 150, # samples
    'numcep': 128, # number of cepstrum bins to return
    'n_fft': 512, #fft lenght
    'fmin': 0, #Hz
    'fmax': 12000 #Hz
}

metric = {
    'lad_doa_thresh': 20,
    'average': 'macro'
}

