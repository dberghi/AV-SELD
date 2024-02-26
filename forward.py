#!/usr/bin/python

import argparse, os
from pathlib import Path
from tqdm import tqdm
import core.config as conf
from core.dataloaders import load_data_from_hdf5
from core.AV_SELD import AV_SELD
from utils.cls_compute_seld_results import ComputeSELDResults
import utils.cls_feature_class as cls_feature_class
import utils.utils as utils
# PyTorch libraries and modules
import torch
from torch.utils.data import DataLoader

base_path = conf.input['project_path']



def main():
    test_file = os.path.join(conf.input['feature_path'],
                             'h5py_{}/test_dataset.h5'.format(conf.training_param['visual_encoder_type']))
    scaler_path = os.path.join(conf.input['feature_path'],
                             'h5py_{}/feature_scaler.h5'.format(conf.training_param['visual_encoder_type']))

    ## ----------- Set output dir --------------------------
    output_folder = Path(os.path.join(base_path, 'output/%s/%f' % (args.info, args.lr)))
    output_folder.mkdir(exist_ok=True, parents=True)

    ## ----------- Set device --------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    ## ------------- Get feature scaler -----------------
    if args.normalize:
        mean, std = utils.load_feature_scaler(scaler_path)
    else:
        mean = None
        std = None

    ## ---------- Data loaders -----------------
    dev_test_cls = cls_feature_class.FeatureClass(train_or_test='test')
    test_seqs_paths_list = dev_test_cls.get_sequences_paths_list()
    d_test = load_data_from_hdf5(test_file, normalize=args.normalize, mean=mean, std=std)
    # keep batch_size=1, shuffle=false, num_workers=1!
    dl_test = DataLoader(d_test, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

    ## ---------- Load model weights -------------
    model = AV_SELD(device=device, **vars(args))
    model.load_weights(os.path.join(base_path, 'ckpt/%s/%f/model_%03d.ckpt' % (args.info, args.lr, args.epoch)))

    # Initialize evaluation metric class
    score_obj = ComputeSELDResults()

    ## ----------- FORWARD TEST SET -------------------------------
    print('Dumping recording-wise test results in: {}'.format(output_folder))
    test_loss = model.test_model(dl_test, output_folder)

    ## ------------ EVALUATION ------------------------------------
    use_jackknife = False # set False for faster evaluation
    test_ER, test_F, test_LE, test_LR, test_seld_scr, classwise_test_scr = score_obj.get_SELD_Results(
        output_folder, is_jackknife=use_jackknife)
    print('\nTest Loss')
    print('SELD score (early stopping metric): {:0.2f} {}'.format(
        test_seld_scr[0] if use_jackknife else test_seld_scr,
        '[{:0.2f}, {:0.2f}]'.format(test_seld_scr[1][0], test_seld_scr[1][1]) if use_jackknife else ''))
    print(
        'SED metrics: Error rate: {:0.2f} {}, F-score: {:0.1f} {}'.format(test_ER[0] if use_jackknife else test_ER,
                                                                          '[{:0.2f}, {:0.2f}]'.format(test_ER[1][0],
                                                                                                      test_ER[1][
                                                                                                          1]) if use_jackknife else '',
                                                                          100 * test_F[
                                                                              0] if use_jackknife else 100 * test_F,
                                                                          '[{:0.2f}, {:0.2f}]'.format(
                                                                              100 * test_F[1][0], 100 * test_F[1][
                                                                                  1]) if use_jackknife else ''))
    print('DOA metrics: Localization error: {:0.1f} {}, Localization Recall: {:0.1f} {}'.format(
        test_LE[0] if use_jackknife else test_LE,
        '[{:0.2f} , {:0.2f}]'.format(test_LE[1][0], test_LE[1][1]) if use_jackknife else '',
        100 * test_LR[0] if use_jackknife else 100 * test_LR,
        '[{:0.2f}, {:0.2f}]'.format(100 * test_LR[1][0], 100 * test_LR[1][1]) if use_jackknife else ''))
    if conf.metric['average'] == 'macro':
        print('Classwise results on unseen test data')
        print('Class\tER\tF\tLE\tLR\tSELD_score')
        for cls_cnt in range(conf.input['num_classes']):
            print('{}\t{:0.2f} {}\t{:0.3f} {}\t{:0.2f} {}\t{:0.3f} {}\t{:0.3f} {}'.format(
                cls_cnt,
                classwise_test_scr[0][0][cls_cnt] if use_jackknife else classwise_test_scr[0][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][0][cls_cnt][0],
                                            classwise_test_scr[1][0][cls_cnt][1]) if use_jackknife else '',
                classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][1][cls_cnt][0],
                                            classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '',
                classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][2][cls_cnt][0],
                                            classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '',
                classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][3][cls_cnt][0],
                                            classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '',
                classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][4][cls_cnt][0],
                                            classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else ''))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test set forward pass, specify arguments')
    parser.add_argument('--epoch', type=int, default=2, metavar='N',
                        help='number of epochs (default: %d)' % conf.training_param['epochs'])
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: %f)' % conf.training_param['learning_rate'])
    parser.add_argument('--normalize', default=True, action='store_true',
                        help='set True to normalize dataset with mean and std of train set')
    parser.add_argument('--info', type=str, default='my_fantastic_model', metavar='S',
                        help='Add additional info for storing')
    parser.add_argument('--model-type', type=str, default=conf.training_param['model_type'], metavar='WS')
    args = parser.parse_args()
    main()