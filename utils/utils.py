#!/usr/bin/python
import csv
import random, h5py, os, cv2

import numpy as np
import soundfile as sf
from PIL import Image
from torchvision import transforms
import core.config as conf
from pathlib import Path
from contextlib import suppress
from sklearn import preprocessing
from utils.SELD_evaluation_metrics import distance_between_cartesian_coordinates





def csv_to_list(csv_path):
    as_list = None
    with open(csv_path, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        as_list = list(reader)
    return as_list

def pad_audio_clip(audio, desired_length): # temporal pad
    """
    Check audio length is as desired_length
    """
    if len(audio) < desired_length: # signal too short
        return np.concatenate((audio,np.zeros((desired_length - len(audio), 4))))
    else: # signal correct length (or for some reasons too long)
        return audio[0: desired_length]


def pad_video_clip(frame_stack, desired_length=30): # temporal pad
    """
    Check if the number of video frames are as desired_length
    """
    t, h, w, c, = frame_stack.shape
    if t < desired_length: # signal too short
        black_frames = np.zeros([desired_length-t,h,w,c]) - 1 # -1 because frames are normalized from -1 to +1
        return np.concatenate((frame_stack, black_frames), axis=0)
    else: # signal correct length (or for some reasons too long)
        return frame_stack[0:desired_length,:,:,:]


def load_rgb_frames(frame_dir, start_time, num=conf.training_param['num_video_frames']): # this is used for I3D
    """
        Load stack of rgb frames

        INPUTS:
          frame_dir: path to video frame directory
          start_time: initial frame in sec
          num: number of frames to load (t)

        OUTPUTS:
          stack of rgb frames (t,h,w,c)
    """
    start_frame = int(round(start_time * conf.input['fps']))
    frames = []
    for i in range(start_frame, start_frame + num):
        if os.path.exists(os.path.join(frame_dir, str(i).zfill(5)+'.jpg')):  # check if file exists.
            img = cv2.imread(os.path.join(frame_dir, str(i).zfill(5)+'.jpg'))[:, :, [2, 1, 0]]
            h,w,c = img.shape
            if w != 448 or h != 224:
                img = cv2.resize(img, dsize=(224, 448))
            img = (img/255.)*2 - 1 # pixel values between -1 and +1
            frames.append(img)
        else: # e.g. end of the sequence
            break
    if frames == []:
        raise ValueError("""Frames not found in '%s' start number %d""" %(frame_dir, start_frame))
    return pad_video_clip(np.asarray(frames, dtype=np.float32), num)


def load_rgb_frames_PIL(frame_dir, start_time, num=conf.training_param['num_video_frames']): # this is used for resnet
    """
        Load stack of rgb frames

        INPUTS:
          frame_dir: path to video frame directory
          start_time: initial frame in sec
          num: number of frames to load (t)

        OUTPUTS:
          stack of rgb frames (t,h,w,c)
    """
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    start_frame = int(round(start_time * conf.input['fps']))
    frames = []
    for i in range(start_frame, start_frame + num):
        if os.path.exists(os.path.join(frame_dir, str(i).zfill(5)+'.jpg')):  # check if file exists.
            img = Image.open(os.path.join(frame_dir, str(i).zfill(5)+'.jpg'))
            img = preprocess(img)
            c, h, w = img.shape
            #img = cv2.imread(os.path.join(frame_dir, str(i).zfill(5)+'.jpg'))[:, :, [2, 1, 0]]
            #if w != 448 or h != 224:
            #    img = cv2.resize(img, dsize=(224, 448))
            #img = (img/255.)*2 - 1 # pixel values between -1 and +1
            frames.append(img)
        else: # e.g. end of the sequence
            break
    if frames == []:
        raise ValueError("""Frames not found in '%s' start number %d""" %(frame_dir, start_frame))
    return frames


def load_audio_file(audio_dir, start_time):
    num_samples = conf.training_param['frame_len_samples'] # number of samples to produce an audio frame (chunk)
    # initial audio sample
    first_audio_sample = int((np.round(start_time * conf.input['label_resolution']) / conf.input['label_resolution']) * conf.input['fs'])

    audio, sr = sf.read(audio_dir)
    # Extract only the chunk
    audio_chunk = audio[first_audio_sample:first_audio_sample + num_samples]
    audio_chunk = pad_audio_clip(audio_chunk, num_samples) # pad in case the extracted segment is too short

    return audio_chunk, sr

def swap_audio_channels(audio, ACS_case):
    """
    audio: multichannel audio (t,c)
    ACS_case: required transformation
    """
    new_audio = np.zeros((audio.shape[0],4))
    if ACS_case == 1:
        new_audio[:, 0] = audio[:, 0]
        new_audio[:, 1] = -audio[:, 3]
        new_audio[:, 2] = -audio[:, 2]
        new_audio[:, 3] = audio[:, 1]
    elif ACS_case == 2:
        new_audio[:, 0] = audio[:, 0]
        new_audio[:, 1] = -audio[:, 3]
        new_audio[:, 2] = audio[:, 2]
        new_audio[:, 3] = -audio[:, 1]
    elif ACS_case == 3:
        new_audio[:, 0] = audio[:, 0]
        new_audio[:, 1] = audio[:, 1]
        new_audio[:, 2] = audio[:, 2]
        new_audio[:, 3] = audio[:, 3]
    elif ACS_case == 4:
        new_audio[:, 0] = audio[:, 0]
        new_audio[:, 1] = -audio[:, 1]
        new_audio[:, 2] = -audio[:, 2]
        new_audio[:, 3] = audio[:, 3]
    elif ACS_case == 5:
        new_audio[:, 0] = audio[:, 0]
        new_audio[:, 1] = audio[:, 3]
        new_audio[:, 2] = -audio[:, 2]
        new_audio[:, 3] = -audio[:, 1]
    elif ACS_case == 6:
        new_audio[:, 0] = audio[:, 0]
        new_audio[:, 1] = audio[:, 3]
        new_audio[:, 2] = audio[:, 2]
        new_audio[:, 3] = audio[:, 1]
    elif ACS_case == 7:
        new_audio[:, 0] = audio[:, 0]
        new_audio[:, 1] = -audio[:, 1]
        new_audio[:, 2] = audio[:, 2]
        new_audio[:, 3] = -audio[:, 3]
    elif ACS_case == 8:
        new_audio[: ,0] = audio[:, 0]
        new_audio[:, 1] = audio[:, 1]
        new_audio[:, 2] = -audio[:, 2]
        new_audio[:, 3] = -audio[:, 3]
    return new_audio



def norm(x, mean, std):
    return (x - mean) / std


def get_latest_ckpt(path, reverse=False, suffix='.ckpt'):
    """
    Load latest checkpoint from target directory. Return None if no checkpoints are found.
    """
    path, file = Path(path), None
    files = (f for f in sorted(path.iterdir(), reverse=not reverse) if f.suffix == suffix)
    with suppress(StopIteration):
        file = next(f for f in files)
    return file




def n_fold_generator(dataset_list, fold_num=5):
    size = len(dataset_list)
    random.shuffle(dataset_list)
    folded_dataset = list()
    fold_size = size // fold_num
    for i in range(0, size, fold_size):
        folded_dataset.append(dataset_list[i:i + fold_size])

    return folded_dataset


def belong_to_val(x, fold):
    condition_true = False
    for i in range(len(fold)):
        if x == fold[i]:
            condition_true = True
    return condition_true


def compute_scaler(h5_file, h5py_dir) -> None:
    """
    Credits to Tho Nguyen et al.
    Adapted from https://github.com/thomeou/SALSA

    Compute feature mean and std vectors of spectrograms for normalization.
    :param h5file_dir: Feature directory that contains train and test folder.
    """
    # Get the dimensions of feature by reading one feature files
    afeature = h5_file['audio_feat'][0]  # (n_channels, n_timesteps, n_features)

    n_channels = afeature.shape[0]
    n_feature_channels = n_channels

    # initialize scaler
    scaler_dict = {}
    for i_chan in np.arange(n_feature_channels):
        scaler_dict[i_chan] = preprocessing.StandardScaler()

    # Iterate through data
    for idx in range(h5_file['audio_feat'].shape[0] // 1):
        afeature = h5_file['audio_feat'][idx]  # (n_channels, n_timesteps, n_features)
        for i_chan in range(n_feature_channels):
            scaler_dict[i_chan].partial_fit(afeature[i_chan, :, :])  # (n_timesteps, n_features)

    # Extract mean and std
    feature_mean = []
    feature_std = []
    for i_chan in range(n_feature_channels):
        feature_mean.append(scaler_dict[i_chan].mean_)
        feature_std.append(np.sqrt(scaler_dict[i_chan].var_))

    feature_mean = np.array(feature_mean)
    feature_std = np.array(feature_std)
    # Expand dims for timesteps: (n_chanels, n_timesteps, n_features)
    feature_mean = np.expand_dims(feature_mean, axis=1)
    feature_std = np.expand_dims(feature_std, axis=1)
    # Save scaler file
    scaler_path = os.path.join(str(h5py_dir) + '/feature_scaler.h5')
    with h5py.File(scaler_path, 'w') as hf:
        hf.create_dataset('mean', data=feature_mean, dtype=np.float32)
        hf.create_dataset('std', data=feature_std, dtype=np.float32)

    print('Scaler path: {}'.format(scaler_path))


def load_feature_scaler(h5_path):
    """
    Load feature scaler for multichannel spectrograms
    """
    with h5py.File(h5_path, 'r') as hf:
        mean = hf['mean'][:]
        std = hf['std'][:]
    return mean, std


def get_multi_accdoa_labels(accdoa_in, nb_classes):
    """
    Args:
        accdoa_in:  [batch_size, frames, num_track*num_axis*num_class=3*3*12]
        nb_classes: scalar
    Return:
        sedX:       [batch_size, frames, num_class=12]
        doaX:       [batch_size, frames, num_axis*num_class=3*12]
    """
    x0, y0, z0 = accdoa_in[:, :, :1 * nb_classes], accdoa_in[:, :, 1 * nb_classes:2 * nb_classes], accdoa_in[:, :,
                                                                                                   2 * nb_classes:3 * nb_classes]
    sed0 = np.sqrt(x0 ** 2 + y0 ** 2 + z0 ** 2) > 0.5
    doa0 = accdoa_in[:, :, :3 * nb_classes]

    x1, y1, z1 = accdoa_in[:, :, 3 * nb_classes:4 * nb_classes], accdoa_in[:, :,
                                                                 4 * nb_classes:5 * nb_classes], accdoa_in[:, :,
                                                                                                 5 * nb_classes:6 * nb_classes]
    sed1 = np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2) > 0.5
    doa1 = accdoa_in[:, :, 3 * nb_classes: 6 * nb_classes]

    x2, y2, z2 = accdoa_in[:, :, 6 * nb_classes:7 * nb_classes], accdoa_in[:, :,
                                                                 7 * nb_classes:8 * nb_classes], accdoa_in[:, :,
                                                                                                 8 * nb_classes:]
    sed2 = np.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2) > 0.5
    doa2 = accdoa_in[:, :, 6 * nb_classes:]

    return sed0, doa0, sed1, doa1, sed2, doa2


def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])


def write_output_format_file(_output_format_file, _output_format_dict):
    """
    Writes DCASE output format csv file, given output format dictionary

    :param _output_format_file:
    :param _output_format_dict:
    :return:
    """
    # open csv with append 'a' attribute if file exists, else create (write) it 'w'
    w_or_a = 'a' if os.path.exists(_output_format_file) else 'w'

    _fid = open(_output_format_file, w_or_a)
    # _fid.write('{},{},{},{}\n'.format('frame number with 20ms hop (int)', 'class index (int)', 'azimuth angle (int)', 'elevation angle (int)'))
    for _frame_ind in _output_format_dict.keys():
        for _value in _output_format_dict[_frame_ind]:
            # Write Cartesian format output. Since baseline does not estimate track count and distance we use fixed values.
            _fid.write('{},{},{},{},{},{},{}\n'.format(int(_frame_ind), int(_value[0]), 0, float(_value[1]), float(_value[2]), float(_value[3]), 0))
    _fid.close()


def determine_similar_location(sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify=15, nb_classes=13):
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        if distance_between_cartesian_coordinates(doa_pred0[class_cnt], doa_pred0[class_cnt + 1 * nb_classes],
                                                  doa_pred0[class_cnt + 2 * nb_classes],
                                                  doa_pred1[class_cnt], doa_pred1[class_cnt + 1 * nb_classes],
                                                  doa_pred1[class_cnt + 2 * nb_classes]) < thresh_unify:
            return 1
        else:
            return 0
    else:
        return 0

def print_stats(output_folder, lr, epoch_cnt, val_ER, val_F, val_LE, val_LR, val_seld_scr, best_val_epoch, best_ER,
                best_F, best_LE, best_LR, best_seld_scr):
    print(
        'epoch: {},'
        'ER/F/LE/LR/SELD: {}, '
        'best_val_epoch: {} {}'.format(epoch_cnt,
                                       '{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}'.format(val_ER, val_F, val_LE, val_LR,
                                                                                        val_seld_scr),
                                       best_val_epoch,
                                       '({:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f})'.format(best_ER, best_F, best_LE,
                                                                                          best_LR,
                                                                                          best_seld_scr))
    )
    print(
        'epoch: {},    '
        'ER/F/LE/LR/SELD: {},    '
        'best_val_epoch: {} {}'.format(epoch_cnt,
                                       '{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}'.format(val_ER, val_F, val_LE, val_LR,
                                                                                        val_seld_scr),
                                       best_val_epoch,
                                       '({:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f})'.format(best_ER, best_F, best_LE,
                                                                                          best_LR,
                                                                                          best_seld_scr)),
        file=open('{}/log_{}.txt'.format(output_folder, lr), "a"))