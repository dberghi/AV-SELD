#!/usr/bin/python

import numpy as np
import argparse, random, glob, os, h5py, time
from pathlib import Path
from tqdm import tqdm
import core.config as conf
from core.dataloaders import load_data_from_hdf5
from core.AV_SELD import AV_SELD
import utils.utils as utils
from utils.cls_compute_seld_results import ComputeSELDResults
import utils.cls_feature_class as cls_feature_class
# PyTorch libraries and modules
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split
# DDP modules
from torch.utils.data.distributed import DistributedSampler



base_path = conf.input['project_path']



def main():

    train_file = os.path.join(conf.input['feature_path'],
                              'h5py_{}/train_dataset.h5'.format(conf.training_param['visual_encoder_type']))
    scaler_path = os.path.join(conf.input['feature_path'],
                              'h5py_{}/feature_scaler.h5'.format(conf.training_param['visual_encoder_type']))
    test_file = os.path.join(conf.input['feature_path'],
                              'h5py_{}/test_dataset.h5'.format(conf.training_param['visual_encoder_type']))

    ## ---------- Experiment reproducibility --------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ## ------------ Set check point directory ----------------
    ckpt_dir = Path(os.path.join(args.ckpt_dir, '%s/%f' % (args.info, args.lr)))
    output_folder = Path(os.path.join(base_path, 'output/%s/%f' % (args.info, args.lr)))
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    output_folder.mkdir(exist_ok=True, parents=True)

    ## --------------- Set device --------------------------
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    #print(device, file=open('%s/log.txt' % ckpt_dir, "a"))

    ## ------------- Get feature scaler -----------------
    if args.normalize:
        mean, std = utils.load_feature_scaler(scaler_path)
    else:
        mean = None
        std = None

    ## ------------- Data loaders -----------------
    train_set = load_data_from_hdf5(train_file, normalize=args.normalize, mean=mean, std=std)
    test_set = load_data_from_hdf5(test_file, normalize=args.normalize, mean=mean, std=std)

    # to split the train set in test-val partitions:
    #t_size = int(0.8 * len(full_train_set)) # 80% for train
    #v_size = len(full_train_set) - t_size # 20% for val
    #train_set, val_set = random_split(train_dataset, [t_size, v_size])


    dl_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    #dl_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_test = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True) # infer one sample at a time

    model = AV_SELD(device=device, **vars(args))

    ## ---------- Look for previous check points -------------
    if args.resume_training:
        ckpt_file = utils.get_latest_ckpt(ckpt_dir)
        if ckpt_file:
            model.load_weights(ckpt_file)
            first_epoch = int(str(ckpt_file)[-8:-5]) + 1
            print('Resuming training from epoch %d' % first_epoch)
            #print('Resuming training from epoch %d' % first_epoch, file=open('%s/log.txt' % ckpt_dir, "a"))
        else:
            print('No checkpoint found in "{}"...'.format(ckpt_dir))
            #print('No checkpoint found in "{}"...'.format(ckpt_dir), file=open('%s/log.txt' % ckpt_dir, "a"))
            first_epoch = 1
    else:
        print('Resume training deactivated')
        #print('Resume training deactivated', file=open('%s/log.txt' % ckpt_dir, "a"))
        first_epoch = 1


    plot_dir = Path(os.path.join(base_path, 'output/training_plots/%s/%f' % (args.info, args.lr)))
    plot_dir.mkdir(exist_ok=True, parents=True)
    if first_epoch == 1:
        train_loss = []
        test_loss = []
        #if args.fold_bool: val_loss = []
    else: # training resumed
        train_loss = torch.load(os.path.join(plot_dir, 'train_loss.pt'))
        test_loss = torch.load(os.path.join(plot_dir, 'test_loss.pt'))

    ## ----------- Initialize evaluation metric class -------------
    score_obj = ComputeSELDResults()
    # start training
    best_val_epoch = -1
    best_ER, best_F, best_LE, best_LR, best_seld_scr = 1., 0., 180., 0., 9999


    ## --------------- TRAIN ------------------------
    for epoch in range(first_epoch, args.epochs+1):
        start_time = time.time()
        train_loss.append(model.train_model(dl_train, epoch, ckpt_dir))
        train_time = time.time() - start_time
        print('Train epoch time: {:0.2f}'.format(train_time))

        if epoch % args.validate_every == 0:
            print('Epoch {}, test forward pass...'.format(epoch))
            #print('Epoch {}, test forward pass...'.format(epoch), file=open('%s/val_log.txt' % ckpt_dir, "a"))
            start_time = time.time()
            ts = model.test_model(dl_test, output_folder)
            #vl = model.validate_model(dl_val, ckpt_dir)
            test_loss.append(ts)

            # Calculate the DCASE 2021 metrics - Location-aware detection and Class-aware localization scores
            val_ER, val_F, val_LE, val_LR, val_seld_scr, classwise_val_scr = score_obj.get_SELD_Results(
                output_folder)

            if val_seld_scr <= best_seld_scr:
                best_val_epoch, best_ER, best_F, best_LE, best_LR, best_seld_scr = epoch, val_ER, val_F, val_LE, val_LR, val_seld_scr

            utils.print_stats(output_folder.parent.absolute(), args.lr, epoch, val_ER, val_F, val_LE, val_LR,
                              val_seld_scr, best_val_epoch, best_ER, best_F, best_LE, best_LR, best_seld_scr)

            test_time = time.time() - start_time
            print('Test epoch time: {:0.2f}'.format(test_time))
            # save test_loss list
            torch.save(test_loss, plot_dir / 'test_loss.pt')

        # save train_loss list
        torch.save(train_loss, plot_dir / 'train_loss.pt')




if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Parse arguments for training')
    parser.add_argument('--batch-size', type=int, default=conf.training_param['batch_size'], metavar='N',
                        help='input batch size for training (default: %d)' % conf.training_param['batch_size'])
    parser.add_argument('--epochs', type=int, default=conf.training_param['epochs'], metavar='N',
                        help='number of epochs to train (default: %d)' % conf.training_param['epochs'])
    parser.add_argument('--lr', type=float, default=conf.training_param['learning_rate'], metavar='LR',
                        help='learning rate (default: %f)' % conf.training_param['learning_rate'])
    parser.add_argument('--normalize', default=True, action='store_true',
                        help='set True to normalize dataset with mean and std of train set')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--validate-every', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--ckpt-dir', default=os.path.join(conf.input['project_path'], 'ckpt'),
                        help='path to save models')
    parser.add_argument('--resume-training', default=True, action='store_true',
                        help='resume training from latest checkpoint')
    parser.add_argument('--info', type=str, default='my_fantastic_model', metavar='S',
                        help='Add additional info for storing')
    parser.add_argument('--fixAudioWeights', default=True, metavar='FxAW',
                        help='whether or not to freeze the audio weights')
    args = parser.parse_args()


    main()
