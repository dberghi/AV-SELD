#!/usr/bin/python
import os.path
import os, glob

import torch
import torch.nn as nn
from models.AV_SELD_model import AudioVisualCMAF, AudioVisualConf, MSELoss_ADPIT
import core.config as conf
import utils.utils as utils
#from torch.nn.parallel import DistributedDataParallel as DDP





class AV_SELD(nn.Module):
    def __init__(self, device, **kwargs):
        super(AV_SELD, self).__init__()
        self.device = device
        self.args = kwargs
        if conf.training_param['model_type'] == 'cmaf':
            self.model = AudioVisualCMAF(device).to(device)
        elif conf.training_param['model_type'] == 'conformer':
            self.model = AudioVisualConf(device).to(device)
        self.opt = conf.training_param['optimizer'](self.model.parameters(), lr=self.args.get('lr'))
        self.criterion = MSELoss_ADPIT()


    def train_model(self, dl_train, epoch, ckpt_dir):
        self.model.train()

        # update learning rate
        if epoch > 30:
            for param_group in self.opt.param_groups:
                param_group['lr'] *= (0.95 ** (epoch - 30)) # 5% decay every epoch after first 30

        training_loss = 0
        for batch_idx, (audio_features, visual_features, labels, initial_time, sequence) in enumerate(dl_train):
            # pass data to gpu (or generally the target device)
            audio_features, visual_features, labels = audio_features.to(self.device), visual_features.to(
                self.device), labels.to(self.device)

            self.opt.zero_grad()  # Clear the gradients if exists
            # forward audio-visual data
            out = self.model(audio_features, visual_features)

            loss = self.criterion(out, labels)

            loss.backward()  # Backpropagate the losses
            self.opt.step()  # Update Model parameters

            if batch_idx % self.args.get('log_interval') == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(audio_features), len(dl_train.dataset),
                           100. * batch_idx / len(dl_train), loss.item() / len(audio_features)))
                #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #    epoch, batch_idx * len(audio_features), len(dl_train.dataset),
                #           100. * batch_idx / len(dl_train), loss.item() / len(audio_features)),
                #    file=open('%s/log.txt' % ckpt_dir, "a"))

            training_loss += loss.item()

            torch.save(self.model.state_dict(), ckpt_dir / 'model_{:03d}.ckpt'.format(epoch))

            #if batch_idx == 40: # for debugging
            #    break

        return training_loss / batch_idx #training_loss / len(dl_train.dataset)  # mean_batch_loss


    def test_model(self, dl_test, output_folder): # feed forward iterating through test set
        self.eval()
        test_loss = 0
        # remove outputs from previous epoch
        if not glob.glob(os.path.join(output_folder, '*')) == []:
            for f in glob.glob(os.path.join(output_folder, '*')): os.remove(f)

        with torch.no_grad():
            for batch_idx, (audio_features, visual_features, target, initial_time, sequence) in enumerate(dl_test):
                # load one batch of data
                audio_features, visual_features, target = audio_features.to(self.device), visual_features.to(
                    self.device), target.to(self.device)

                # process the batch of data based on chosen mode
                output = self.model(audio_features, visual_features)
                loss = self.criterion(output, target)

                sed_pred0, doa_pred0, sed_pred1, doa_pred1, sed_pred2, doa_pred2 = utils.get_multi_accdoa_labels(
                    output.detach().cpu().numpy(), conf.input['num_classes'])
                sed_pred0 = utils.reshape_3Dto2D(sed_pred0)
                doa_pred0 = utils.reshape_3Dto2D(doa_pred0)
                sed_pred1 = utils.reshape_3Dto2D(sed_pred1)
                doa_pred1 = utils.reshape_3Dto2D(doa_pred1)
                sed_pred2 = utils.reshape_3Dto2D(sed_pred2)
                doa_pred2 = utils.reshape_3Dto2D(doa_pred2)

                # dump SELD results to the corresponding file
                output_file = os.path.join(output_folder, '{}{}'.format(sequence[0], '.csv'))
                #file_cnt += 1
                output_dict = {}

                for frame_cnt in range(sed_pred0.shape[0]):
                    frame_number = frame_cnt + int(initial_time * conf.input['label_resolution'])
                    for class_cnt in range(sed_pred0.shape[1]):
                        # determine whether track0 is similar to track1
                        flag_0sim1 = utils.determine_similar_location(sed_pred0[frame_cnt][class_cnt],
                                                                sed_pred1[frame_cnt][class_cnt],
                                                                doa_pred0[frame_cnt],
                                                                doa_pred1[frame_cnt], class_cnt,
                                                                thresh_unify=15,
                                                                nb_classes=conf.input['num_classes'])
                        flag_1sim2 = utils.determine_similar_location(sed_pred1[frame_cnt][class_cnt],
                                                                sed_pred2[frame_cnt][class_cnt],
                                                                doa_pred1[frame_cnt],
                                                                doa_pred2[frame_cnt], class_cnt,
                                                                thresh_unify=15,
                                                                nb_classes=conf.input['num_classes'])
                        flag_2sim0 = utils.determine_similar_location(sed_pred2[frame_cnt][class_cnt],
                                                                sed_pred0[frame_cnt][class_cnt],
                                                                doa_pred2[frame_cnt],
                                                                doa_pred0[frame_cnt], class_cnt,
                                                                thresh_unify=15,
                                                                nb_classes=conf.input['num_classes'])
                        # unify or not unify according to flag
                        if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                            if sed_pred0[frame_cnt][class_cnt] > 0.5:
                                if frame_number not in output_dict:
                                    output_dict[frame_number] = []
                                output_dict[frame_number].append([class_cnt, doa_pred0[frame_cnt][class_cnt],
                                                               doa_pred0[frame_cnt][
                                                                   class_cnt + conf.input['num_classes']],
                                                               doa_pred0[frame_cnt][
                                                                   class_cnt + 2 * conf.input['num_classes']]])
                            if sed_pred1[frame_cnt][class_cnt] > 0.5:
                                if frame_number not in output_dict:
                                    output_dict[frame_number] = []
                                output_dict[frame_number].append([class_cnt, doa_pred1[frame_cnt][class_cnt],
                                                               doa_pred1[frame_cnt][
                                                                   class_cnt + conf.input['num_classes']],
                                                               doa_pred1[frame_cnt][
                                                                   class_cnt + 2 * conf.input['num_classes']]])
                            if sed_pred2[frame_cnt][class_cnt] > 0.5:
                                if frame_number not in output_dict:
                                    output_dict[frame_number] = []
                                output_dict[frame_number].append([class_cnt, doa_pred2[frame_cnt][class_cnt],
                                                               doa_pred2[frame_cnt][
                                                                   class_cnt + conf.input['num_classes']],
                                                               doa_pred2[frame_cnt][
                                                                   class_cnt + 2 * conf.input['num_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                            if frame_number not in output_dict:
                                output_dict[frame_number] = []
                            if flag_0sim1:
                                if sed_pred2[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_number].append([class_cnt, doa_pred2[frame_cnt][class_cnt],
                                                                   doa_pred2[frame_cnt][
                                                                       class_cnt + conf.input['num_classes']],
                                                                   doa_pred2[frame_cnt][
                                                                       class_cnt + 2 * conf.input['num_classes']]])
                                doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                                output_dict[frame_number].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + conf.input['num_classes']],
                                                               doa_pred_fc[
                                                                   class_cnt + 2 * conf.input['num_classes']]])
                            elif flag_1sim2:
                                if sed_pred0[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_number].append([class_cnt, doa_pred0[frame_cnt][class_cnt],
                                                                   doa_pred0[frame_cnt][
                                                                       class_cnt + conf.input['num_classes']],
                                                                   doa_pred0[frame_cnt][
                                                                       class_cnt + 2 * conf.input['num_classes']]])
                                doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                                output_dict[frame_number].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + conf.input['num_classes']],
                                                               doa_pred_fc[
                                                                   class_cnt + 2 * conf.input['num_classes']]])
                            elif flag_2sim0:
                                if sed_pred1[frame_cnt][class_cnt] > 0.5:
                                    output_dict[frame_number].append([class_cnt, doa_pred1[frame_cnt][class_cnt],
                                                                   doa_pred1[frame_cnt][
                                                                       class_cnt + conf.input['num_classes']],
                                                                   doa_pred1[frame_cnt][
                                                                       class_cnt + 2 * conf.input['num_classes']]])
                                doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                                output_dict[frame_number].append([class_cnt, doa_pred_fc[class_cnt],
                                                               doa_pred_fc[class_cnt + conf.input['num_classes']],
                                                               doa_pred_fc[
                                                                   class_cnt + 2 * conf.input['num_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                            if frame_number not in output_dict:
                                output_dict[frame_number] = []
                            doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                            output_dict[frame_number].append(
                                [class_cnt, doa_pred_fc[class_cnt],
                                 doa_pred_fc[class_cnt + conf.input['num_classes']],
                                 doa_pred_fc[class_cnt + 2 * conf.input['num_classes']]])

                utils.write_output_format_file(output_file, output_dict)

                test_loss += loss.item()
                #nb_test_batches += 1

            mean_sample_loss = test_loss / len(dl_test.dataset)
        return mean_sample_loss # consider that batch=1 in test pass


    def validate_model(self, dl_val):
        self.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_idx, (audio_features, visual_features, target, initial_time, sequence) in enumerate(dl_val):
                # load one batch of data
                audio_features, visual_features, target = audio_features.to(self.device), visual_features.to(
                    self.device), target.to(self.device)

                # process the batch of data based on chosen mode
                output = self.model(audio_features, visual_features)
                loss = self.criterion(output, target)
                val_loss += loss.item()

            mean_batch_loss = val_loss / batch_idx
        print('Mean batch loss: {}'.format(mean_batch_loss))
        #print('Mean batch loss: {}'.format(mean_batch_loss), file=open('%s/val_log.txt' % ckpt_dir, "a"))
        return mean_batch_loss
    

    def load_weights(self, ckpt_file):
        checkpoint = torch.load(ckpt_file, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint)

    # used to save the audio backbone weights
    def save_audio_backbone(self, output_dir, epoch):
        torch.save(self.model.audioFrontEnd.state_dict(), output_dir / 'audio_{:03d}.ckpt'.format(epoch))


