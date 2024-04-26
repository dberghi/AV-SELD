# Fusion of Audio and Visual Embeddings for Sound Event Localization and Detection


This repository contains the python implementation for the paper "Fusion of Audio and Visual Embeddings for Sound Event Localization and Detection" which has been presented at IEEE ICASSP 2024.
> D. Berghi, P. Wu, J. Zhao, W. Wang, P. J. B. Jackson. Fusion of Audio and Visual Embeddings for Sound Event Localization and Detection. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024. [[**arXiv**]](https://arxiv.org/abs/2312.09034) [[**IEEE Xplore**]](https://ieeexplore.ieee.org/abstract/document/10448050)



## Dataset

- Download the STARSS23 from https://zenodo.org/records/7880637 

- Extract the .zip files in the `data_dcase2023_task3/` folder to get the following directory structure:

<pre>
	.
	└── data_dcase2023_task3
	    ├── foa_dev
	    ├── foa_eval
	    ├── LICENSE
	    ├── list_dataset
	    ├── metadata_dev
	    ├── mic_dev
	    ├── mic_eval
	    ├── README.md
	    ├── video_dev
	    └── video_eval
</pre>

`data_dcase2023_task3/` does not have to be within your project directory, you can move it somewhere else if you'd like. 
As in the paper, this implementation only supports the employment of the development set and the FOA audio format. Therefore, you will only need foa_dev, metadata_dev, video_dev, and list_dataset (provided). If desired, this code can be easily extended to the MIC format and the evaluation set. 


## Get started

### Dependencies

Create a new environment and install the dependencies
```
conda create -n AV-SELD python=3.9
conda activate AV-SELD
pip install -r utils/requirement.txt
```

OR install the dependencies from an existing environment
```
pip install -r utils/requirement.txt
```
Alternatively, you can manually install the modules in `./utils/requirements.txt`

### Configurations

Open `core/config.py` to edit the input configurations. Change `project_path` to point to your working directory, and `data_path` to point to the `data_dcase2023_task3/` dataset folder. Change `feature_path` too: this indicates the location where the audio and visual features will be stored. 
In the `training_param` section you can set which visual encoder to adopt between 'resnet' and 'i3d' (row 30), and which fusion mechanism type between 'conformer' and 'cmaf' (row 31). These choices are case-sensitive: other selections or typos won't work.

There are 4 bash files in the `scripts/` folder (`0_extract_frames.sh`, `1_preprocessing.sh`, `2_train.sh`, `3_forward.sh`). They are used to call the required python scripts and run the experiments.
NOTE: You might need to make the bash files executable by running `chmod +x scripts/<file_name>.sh`

### Extract frames and apply audio-visual channel swap (AVCS)

This step will first extract the video frames at a resolution of 448x224p (10fps). Then, these frames will undergo 7 transformations aligned with the audio channel swap technique. The so-extracted frames will be stored in a new folder in `data_dcase2023_task3/frame_dev/`. Frames' folders are enumerated from 1 to 8 following the audio channel swap (ACS) rules below.
```
	 1:	φ = φ − pi/2, θ = −θ
         2:	φ = −φ − pi/2, θ = θ
         3:	φ = φ, θ = θ
	 4:	φ = −φ, θ = −θ
         5:	φ = φ + pi/2, θ = −θ
         6:	φ = −φ + pi/2, θ = θ
         7:	φ = φ + pi, θ = θ
         8:	φ = −φ + pi, θ = −θ
```
For example, in folder "3" are extracted the original "untransformed" frames.

To start the frame extraction and AVCS transformation run:

	bash scripts/0_extract_frames.sh

NOTE: this operation will take a few hours to complete and requires ~75 GB of free space. After extracting the frames in folder "3", the bash script applies the 7 transformations, one at a time. 

Optionally, the process can be sped up by parallelizing the transformations. I.e., instead of running the bash script, extract the frames with `python core/extract_frames_and_AVCS.py --ACS-case 3`. **Once it finishes,** perform the 7 transformations in parallel with `python core/video_channel_swap.py --ACS-case <case_number>` and replace <case_number> with 1, 2, 4, 5, 6, 7, and 8 in seven parallel jobs. 


## 1 - EXTRACT INPUT FEATURES 

This step extracts and stores audio and visual features used to train the network.
The audio encoder takes as input intensity vectors features from the FOA audio signals. We also directly generate and save the visual embeddings by processing the video frames with ResNet50 or I3D (working with pre-extracted visual embeddings, instead of re-generating them every time, remarkably speeds up the training). 
The code extracts the intensity vectors and visual embeddings for the train and test partitions and stores them in HDF5 binary format using the h5py module. 
Then it computes the mean and standard deviation vectors that will be used to normalize the audio input features before training (`feature_scaler.h5`).

NOTE: Nearly **500GB of free space is required** to store the extracted features. 
Create h5py datasets by running:
 
	bash scripts/1_preprocessing.sh

This will create a `train_dataset.h5`, `test_dataset.h5`, and `feature_scaler.h5` files and store them in `[feature_path]/features/h5py_[visual_encoder]/`


## 2 - TRAINING

You can customize your training by setting the input string argument "INFO" in `scripts/2_train.sh`. You can also set the number of epochs ("EPOCHS") and learning rate ("LR"). Even though we trained all our models for 50 epochs, they usually converge within the first 10 epochs. Therefore, we recommend setting the number of epochs to 10. Note, the input arguments that you parse from the bash script will override the configuration settings in `core/config.py`.
Start the training by running:

	bash scripts/2_train.sh

The model's weights will be saved in the checkpoint folder `ckpt/[INFO]/[LR]/`.
By default, the model will be trained on the entire training split and evaluated on the test split. At the end of each epoch, the predictions of the test set's sequences will be stored in `output/[INFO]/[LR]` and the SELD metrics printed.


## 3 - FORWARD and EVALUATE TEST SET

To evaluate a pre-trained model you can run:

	bash scripts/3_forward.sh

Set the input arguments of `scripts/3_forward.sh` according to the `[INFO]` of the pre-trained model, the best epoch, and learning rate that you would like to (re-)evaluate. 
This will replicate a forward pass of the test set (as it is done at the end of each training epoch during training).


## Pretrained audio and visual encoders

The CNN-Conformer employed as an audio encoder was pre-trained on SELD. The weights are stored in `models/weights/audio_weights.pt`. The weights for I3D visual encoder are available in `models/weights/rgb_imagenet.pt` (we do not employ the branch pre-trained on optical flow). The weights for the ResNet50 are available with Torchvision.


# Citation

Please consider citing our paper if you find this code useful, many thanks!

```
@INPROCEEDINGS{Berghi:2024:avseld,
  author={Berghi, Davide and Wu, Peipei and Zhao, Jinzheng and Wang, Wenwu and Jackson, Philip J. B.},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing}, 
  title={Fusion of Audio and Visual Embeddings for Sound Event Localization and Detection}, 
  year={2024},
  pages={8816-8820},
  doi={10.1109/ICASSP48485.2024.10448050}
}
```

## Acknowledge

This repository adapts and integrates scripts from the original repo provided for the DCASE Task 3 Challenge baseline (audio-only) and other repos. In particular:

* The prediction of multi-ACCDOA vectors, the ADPIT loss function, and the implementation of the SELD metrics can be found at: https://github.com/sharathadavanne/seld-dcase2023 
* The extraction of intensity vectors for the FOA audio format is adapted from Yin Cao's https://github.com/yinkalario/EIN-SELD
* The Pytorch implementation of I3D and the respective pre-trained models can be found in AJ Piergiovanni's https://github.com/piergiaj/pytorch-i3d/tree/master
