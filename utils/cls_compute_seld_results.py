"""
This class script is taken from the DCASE Task 3 baseline code.
Credits to Adavanne et al. Available from:
https://github.com/sharathadavanne/seld-dcase2023
"""

import os
import utils.SELD_evaluation_metrics as SELD_evaluation_metrics
import utils.cls_feature_class as cls_feature_class
#import parameters
import numpy as np
from scipy import stats
import core.config as conf


def jackknife_estimation(global_value, partial_estimates, significance_level=0.05):
    """
    Compute jackknife statistics from a global value and partial estimates.
    Original function by Nicolas Turpault

    :param global_value: Value calculated using all (N) examples
    :param partial_estimates: Partial estimates using N-1 examples at a time
    :param significance_level: Significance value used for t-test

    :return:
    estimate: estimated value using partial estimates
    bias: Bias computed between global value and the partial estimates
    std_err: Standard deviation of partial estimates
    conf_interval: Confidence interval obtained after t-test
    """

    mean_jack_stat = np.mean(partial_estimates)
    n = len(partial_estimates)
    bias = (n - 1) * (mean_jack_stat - global_value)

    std_err = np.sqrt(
        (n - 1) * np.mean((partial_estimates - mean_jack_stat) * (partial_estimates - mean_jack_stat), axis=0)
    )

    # bias-corrected "jackknifed estimate"
    estimate = global_value - bias

    # jackknife confidence interval
    if not (0 < significance_level < 1):
        raise ValueError("confidence level must be in (0, 1).")

    t_value = stats.t.ppf(1 - significance_level / 2, n - 1)

    # t-test
    conf_interval = estimate + t_value * np.array((-std_err, std_err))

    return estimate, bias, std_err, conf_interval

class ComputeSELDResults(object):
    def __init__(
            self, ref_files_folder=None, use_polar_format=True
    ):
        self._use_polar_format = use_polar_format
        #self._desc_dir = ref_files_folder if ref_files_folder is not None else os.path.join(conf.input['data_path'], 'metadata_dev')
        self._doa_thresh = conf.metric['lad_doa_thresh'] #params['lad_doa_thresh']

        # Load feature class
        self._feat_cls = cls_feature_class.FeatureClass()

        self._list_dataset, self._dataset_combination = ref_files_folder if ref_files_folder is not None else self._feat_cls.get_list_dataset()
        
        # collect reference files
        self._ref_labels = {}
        '''
        for split in os.listdir(self._desc_dir):      
            for ref_file in os.listdir(os.path.join(self._desc_dir, split)):
                # Load reference description file
                gt_dict = self._feat_cls.load_output_format_file(os.path.join(self._desc_dir, split, ref_file))
                if not self._use_polar_format:
                    gt_dict = self._feat_cls.convert_output_format_polar_to_cartesian(gt_dict)
                nb_ref_frames = max(list(gt_dict.keys()))
                self._ref_labels[ref_file] = [self._feat_cls.segment_labels(gt_dict, nb_ref_frames), nb_ref_frames]
        '''
        txt_file = open(self._list_dataset, 'r')
        rows = [line.rstrip('\n') for line in txt_file]
        for file_cnt, sequence_path in enumerate(rows):
            ref_file = sequence_path.replace(self._dataset_combination, 'metadata_dev').replace('wav', 'csv')
            ref_file = os.path.join(conf.input['data_path'], ref_file)
            # Load reference description file
            gt_dict = self._feat_cls.load_output_format_file(ref_file)
            if not self._use_polar_format:
                gt_dict = self._feat_cls.convert_output_format_polar_to_cartesian(gt_dict)
            nb_ref_frames = max(list(gt_dict.keys()))
            self._ref_labels[os.path.basename(ref_file)] = [self._feat_cls.segment_labels(gt_dict, nb_ref_frames), nb_ref_frames]

        self._nb_ref_files = len(self._ref_labels)
        self._average = conf.metric['average']

    @staticmethod
    def get_nb_files(file_list, tag='all'):
        '''
        Given the file_list, this function returns a subset of files corresponding to the tag.

        Tags supported
        'all' -
        'ir'

        :param file_list: complete list of predicted files
        :param tag: Supports two tags 'all', 'ir'
        :return: Subset of files according to chosen tag
        '''
        _group_ind = {'room': 10}
        _cnt_dict = {}
        for _filename in file_list:

            if tag == 'all':
                _ind = 0
            else:
                _ind = int(_filename[_group_ind[tag]])

            if _ind not in _cnt_dict:
                _cnt_dict[_ind] = []
            _cnt_dict[_ind].append(_filename)

        return _cnt_dict

    def get_SELD_Results(self, pred_files_path, is_jackknife=False):
        # collect predicted files info
        pred_files = os.listdir(pred_files_path)
        pred_labels_dict = {}
        eval = SELD_evaluation_metrics.SELDMetrics(nb_classes=self._feat_cls.get_nb_classes(), doa_threshold=self._doa_thresh, average=self._average)
        for pred_cnt, pred_file in enumerate(pred_files):
            # Load predicted output format file
            pred_dict = self._feat_cls.load_output_format_file(os.path.join(pred_files_path, pred_file))
            if self._use_polar_format:
                pred_dict = self._feat_cls.convert_output_format_cartesian_to_polar(pred_dict)
            pred_labels = self._feat_cls.segment_labels(pred_dict, self._ref_labels[pred_file][1])
            # Calculated scores
            eval.update_seld_scores(pred_labels, self._ref_labels[pred_file][0])
            if is_jackknife:
                pred_labels_dict[pred_file] = pred_labels
        # Overall SED and DOA scores
        ER, F, LE, LR, seld_scr, classwise_results = eval.compute_seld_scores()

        if is_jackknife:
            global_values = [ER, F, LE, LR, seld_scr]
            if len(classwise_results):
                global_values.extend(classwise_results.reshape(-1).tolist())
            partial_estimates = []
            # Calculate partial estimates by leave-one-out method
            for leave_file in pred_files:
                leave_one_out_list = pred_files[:]
                leave_one_out_list.remove(leave_file)
                eval = SELD_evaluation_metrics.SELDMetrics(nb_classes=self._feat_cls.get_nb_classes(), doa_threshold=self._doa_thresh, average=self._average)
                for pred_cnt, pred_file in enumerate(leave_one_out_list):
                    # Calculated scores
                    eval.update_seld_scores(pred_labels_dict[pred_file], self._ref_labels[pred_file][0])
                ER, F, LE, LR, seld_scr, classwise_results = eval.compute_seld_scores()
                leave_one_out_est = [ER, F, LE, LR, seld_scr]
                if len(classwise_results):
                    leave_one_out_est.extend(classwise_results.reshape(-1).tolist())

                # Overall SED and DOA scores
                partial_estimates.append(leave_one_out_est)
            partial_estimates = np.array(partial_estimates)
                    
            estimate, bias, std_err, conf_interval = [-1]*len(global_values), [-1]*len(global_values), [-1]*len(global_values), [-1]*len(global_values)
            for i in range(len(global_values)):
                estimate[i], bias[i], std_err[i], conf_interval[i] = jackknife_estimation(
                           global_value=global_values[i],
                           partial_estimates=partial_estimates[:, i],
                           significance_level=0.05
                           )
            return [ER, conf_interval[0]], [F, conf_interval[1]], [LE, conf_interval[2]], [LR, conf_interval[3]], [seld_scr, conf_interval[4]], [classwise_results, np.array(conf_interval)[5:].reshape(5,13,2) if len(classwise_results) else []]
      
        else:      
            return ER, F, LE, LR, seld_scr, classwise_results


