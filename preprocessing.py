#!/usr/bin/python

import h5py, argparse, os
import torch
from pathlib import Path
from torchvision.models import resnet50, ResNet50_Weights
from models.pytorch_i3d import InceptionI3d
import core.config as conf
from core.dataloaders import load_data_from_scratch
import utils.cls_feature_class as cls_feature_class
from utils.extract_IV import LogmelIntensity_Extractor
from torch.utils.data import DataLoader
from tqdm import tqdm
import utils.utils as utils

base_path = conf.input['project_path']
output_base_path = conf.input['feature_path']




def main():
    dev_train_cls = cls_feature_class.FeatureClass(train_or_test=args.train_or_test)
    # # Extract labels
    dev_train_cls.extract_all_labels()
    dataset_stats = dev_train_cls.get_filewise_frames()
    sequences_paths_list = dev_train_cls.get_sequences_paths_list()
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    #print(device, file=open('log.txt', "a"))

    # audio feature extractor
    feature_extractor = LogmelIntensity_Extractor(conf)
    # video feature extractor(s)
    if conf.training_param['visual_encoder_type'] == 'resnet':
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        # modify final layer
        resnet.avgpool = torch.nn.AvgPool2d(kernel_size=[7, 7], stride=7)
        modules = list(resnet.children())[:-1]
        resnetFeatExtractor = torch.nn.Sequential(*modules)
        resnetFeatExtractor.to(device)
        resnetFeatExtractor.eval()
    elif conf.training_param['visual_encoder_type'] == 'i3d':
        i3dFeatExtractor = InceptionI3d(400, in_channels=3).to(device)
        i3dFeatExtractor.load_state_dict(
        torch.load(os.path.join(base_path, 'models', 'weights', 'rgb_imagenet.pt'), map_location=torch.device(device)))
        i3dFeatExtractor.eval()
    else:
        raise ValueError("""Input feature '%s' non supported. Use 'resnet' or 'i3d' instead.""" % conf.training_param[
            'video_encoder_type'])

    h5py_dir = Path(os.path.join(output_base_path, 'h5py_{}'.format(conf.training_param['visual_encoder_type']))) # output dir
    h5py_dir.mkdir(exist_ok=True, parents=True)

    ## ---------- Data loaders -----------------
    # N.B. here keep normalize=False, batch_size=1, shuffle=False !!!!
    d_dataset = load_data_from_scratch(dataset_stats, sequences_paths_list,
                                       visual_encoder=conf.training_param['visual_encoder_type'],
                                       train_or_test=args.train_or_test)
    data_loader = DataLoader(d_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    #f = h5py.File('{}h5py/{}_dataset.h5'.format(output_base_path, args.train_or_test), 'a')
    f = h5py.File(os.path.join(h5py_dir, '{}_dataset.h5'.format(args.train_or_test)), 'a')

    for count, data_point in enumerate(tqdm(data_loader)):
        # Data to be appended
        audio = data_point[0].float()
        frames = data_point[1] #torch.squeeze(data_point[1], 0)
        #fr_l = data_point[2]
        #fr_r = data_point[3]
        labels = data_point[2]
        sequence = data_point[3]
        initial_time = data_point[4]
        #print('Sequence {}; frames shape: {}'.format(sequence, frames.shape))
        # Extract features
        audio_features = feature_extractor(audio)
        num_frames = conf.input['fps'] * conf.input['input_len_sec']

        if conf.training_param['visual_encoder_type'] == 'resnet':
            with torch.no_grad():
                frames = torch.squeeze(frames, 0).to(device)
                video_embedding = resnetFeatExtractor(frames)
            video_embedding = torch.reshape(video_embedding, (video_embedding.size(0), 4096))
            video_embedding = video_embedding.cpu()
            t, dim = video_embedding.shape
            if t < num_frames:  # e.g. at the end of the sequence. padding required
                video_embedding = torch.cat((video_embedding, torch.zeros((num_frames - t, dim))), dim=0)
            video_embedding = video_embedding.unsqueeze(0)  # create a mini-batch to store hdf5

        elif conf.training_param['visual_encoder_type'] == 'i3d':
            fr_l, fr_r = frames[0], frames[1]
            fr_l, fr_r = fr_l.to(device, dtype=torch.float), fr_r.to(device, dtype=torch.float)
            video_embedding = forward_visual_I3D(i3dFeatExtractor, fr_l, fr_r)
            video_embedding = video_embedding.cpu()
            video_embedding = video_embedding.detach().numpy()
            _, _, dim = video_embedding.shape

        if count == 0:
            # Create the dataset during the first iteration
            f.create_dataset('audio_feat', data=audio_features, chunks=True, maxshape=(None, 7, 480, 128))
            f.create_dataset('video_feat', data=video_embedding, chunks=True, maxshape=(None, num_frames, dim))
            f.create_dataset('labels', data=labels, maxshape=(None, 30, 6, 4, 13), chunks=True)  # , maxshape=(None, 1))
            f.create_dataset('sequences', data=sequence, maxshape=(None,), chunks=True)
            f.create_dataset('initial_time', data=initial_time, maxshape=(None,), chunks=True)
        else:

            f['audio_feat'].resize((f['audio_feat'].shape[0] + audio_features.shape[0]), axis=0)
            f['audio_feat'][-audio_features.shape[0]:] = audio_features

            f['video_feat'].resize((f['video_feat'].shape[0] + video_embedding.shape[0]), axis=0)
            f['video_feat'][-video_embedding.shape[0]:] = video_embedding

            f['labels'].resize((f['labels'].shape[0] + labels.shape[0]), axis=0)
            f['labels'][-labels.shape[0]:] = labels

            f['sequences'].resize((f['sequences'].shape[0] + len(sequence)), axis=0)
            f['sequences'][-len(sequence):] = sequence

            f['initial_time'].resize((f['initial_time'].shape[0] + len(initial_time)), axis=0)
            f['initial_time'][-len(initial_time):] = initial_time


        #print("'audio_feat' chunk has shape:{}".format(f['audio_feat'].shape))
        #print("'audio_feat' chunk has shape:{}".format(f['audio_feat'].shape),
        #      file=open('%s/h5py_log_%s.txt' % (h5py_dir, args.train_or_test), "a"))

        #if count == 200: # used as early stop for debugging
        #    break

    if args.train_or_test == 'train':
        print('============> Computing feature scaler...')
        #print('============> Computing feature scaler...', file=open('%s/make_h5py_log.txt' % h5py_dir, "a"))
        utils.compute_scaler(f, h5py_dir) # compute training set statistics. Will be used later for normalization

    f.close()

def forward_visual_I3D(i3dFeatExtractor, x_l, x_r):
    '''x -> (batch_size, channels, time, height, width)'''
    x_l = i3dFeatExtractor.extract_features(x_l)
    x_r = i3dFeatExtractor.extract_features(x_r)
    x_l = torch.squeeze(torch.squeeze(x_l, 4),
                        3)  # to avoid squeezing the batch dimension when B=1. out:(B, 1024, T)
    x_r = torch.squeeze(torch.squeeze(x_r, 4), 3)
    x_l = x_l.transpose(1, 2)  # (B, T, 1024)
    x_r = x_r.transpose(1, 2)

    x = torch.cat((x_l, x_r), 2)  # (B, T, 512)
    # temporal interpolation (upsample)
    x_interp = torch.nn.functional.interpolate(x.transpose(1, 2),
                                         size=int(conf.input['input_len_sec'] * conf.input['label_resolution']),
                                         mode='nearest')
    x_interp = x_interp.transpose(1, 2)
    return x_interp # (1, T, 2048)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract and save input features')
    parser.add_argument('--train-or-test', type=str, default='test', metavar='S',
                        help='choose train or test partition')
    args = parser.parse_args()
    main()