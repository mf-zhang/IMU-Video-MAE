# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader_old.py

# modified from:
# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch

import csv
import json
import os.path

import pandas as pd
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import torchvision.transforms as T
from PIL import Image
import PIL
import decord
from transforms import GroupNormalize, GroupMultiScaleCrop, Stack, ToTorchFormatTensor

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1

    return index_lookup

def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            name_lookup[row['index']] = row['display_name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        return mask 

class DataAugmentationForVideoMAE(object):
    def __init__(self, args=None, video_masking_ratio=0.9):
        self.input_size = 224
        self.mask_type = 'tube'
        self.window_size = (8,14,14)
        self.mask_ratio = video_masking_ratio

        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])
        self.transform = T.Compose([                            
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if self.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                self.window_size, self.mask_ratio
            )

    def __call__(self, images):
        process_data, _ = self.transform(images)
        # print(len(images), len(images[0]))
        # print(images[0][0].size)
        # print(process_data.shape)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

class EVIDataset(Dataset):
    def __init__(self, dataset_json_file, imu_conf, label_csv=None, video_masking_ratio=0.9, image_as_video=False):
        """
        Dataset that manages imu recordings
        :param imu_conf: Dictionary containing the imu loading and preprocessing settings
        :param dataset_json_file
        """

        self.use_imu = True
        self.datapath = dataset_json_file
        print(self.datapath)
        if 'wear' in self.datapath:
            self.dataset_name = 'wear'
        else:
            self.dataset_name = 'cmu'
        self.data_base_path = os.path.dirname(os.path.dirname(os.path.dirname(self.datapath)))
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        self.data = self.pro_data(self.data)
        print('Dataset has {:d} samples'.format(self.data.shape[0]))
        self.num_samples = self.data.shape[0]
        self.imu_conf = imu_conf
        self.label_smooth = self.imu_conf.get('label_smooth', 0.0)
        print('Using Label Smoothing: ' + str(self.label_smooth))
        self.melbins = self.imu_conf.get('num_mel_bins')
        self.freqm = self.imu_conf.get('freqm', 0)
        self.timem = self.imu_conf.get('timem', 0)
        print('now using following mask: {:d} freq, {:d} time'.format(self.imu_conf.get('freqm'), self.imu_conf.get('timem')))
        self.mixup = self.imu_conf.get('mixup', 0)
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.imu_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.imu_conf.get('mean')
        self.norm_std = self.imu_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.imu_conf.get('skip_norm') if self.imu_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))

        # if add noise for data augmentation
        self.noise = self.imu_conf.get('noise', False)
        if self.noise == True:
            print('now use noise augmentation')
        else:
            print('not use noise augmentation')

        self.index_dict = make_index_dict(label_csv)
        self.label_num = len(self.index_dict)
        print('number of classes is {:d}'.format(self.label_num))

        self.target_length = self.imu_conf.get('target_length')

        # train or eval
        self.mode = self.imu_conf.get('mode')
        print('now in {:s} mode.'.format(self.mode))

        # no use
        # set the frame to use in the eval mode, default value for training is -1 which means random frame
        self.frame_use = self.imu_conf.get('frame_use', -1)
        # by default, 10 frames are used
        self.total_frame = self.imu_conf.get('total_frame', 10)
        # print('now use frame {:d} from total {:d} frames'.format(self.frame_use, self.total_frame))

        # by default, all models use 224*224, other resolutions are not tested
        self.im_res = self.imu_conf.get('im_res', 224) # 224
        print('now using {:d} * {:d} image input'.format(self.im_res, self.im_res))
        self.preprocess = T.Compose([
            T.Resize(self.im_res, interpolation=PIL.Image.BICUBIC),
            T.CenterCrop(self.im_res),
            T.ToTensor(),
            T.Normalize(
                mean=[0.4850, 0.4560, 0.4060],
                std=[0.2290, 0.2240, 0.2250]
            )])
        
        self.spectrogram_transform = torchaudio.transforms.Spectrogram(
            n_fft=256,
            win_length=24,
            hop_length=1,
            window_fn=torch.hann_window
        )

        # for videomae
        self.video_transform = DataAugmentationForVideoMAE(video_masking_ratio=video_masking_ratio)
        self.num_segments = 1
        self.skip_length = 64
        self.new_step = 4
        self.new_length = 16
        assert self.skip_length / self.new_step == self.new_length
        self.temporal_jitter = False

        self.save_visualization_path = None # './visualization_dataloader_wear/' # if None, no visualization

        self.image_as_video = image_as_video

    # change python list to numpy array to avoid memory leak.
    def pro_data(self, data_json):
        for i in range(len(data_json)):
            data_json[i] = [data_json[i]['imu'], data_json[i]['label'], data_json[i]['video_id'], data_json[i]['frame_path']]
        data_np = np.array(data_json, dtype=str)
        return data_np

    def decode_data(self, np_data):
        datum = {}
        if self.use_imu:
            datum['imu'] = np_data[0]
            datum['labels'] = np_data[1]
            datum['video_id'] = np_data[2]
            datum['video_path'] = np_data[3]
        return datum

    def _imu2fbank(self, filename, video_frame_id_list, video_duration, filename2=None, mix_lambda=-1):
        fbank_list = []
        raw_list = []
        if self.dataset_name == 'cmu':
            imu_to_use = [1,8,15,  5,12,19,  3,10,17,  4,11,18] # xyz acceleration for left arm, right arm, left leg, right leg
        elif self.dataset_name == 'wear':
            imu_to_use = [10,11,12, 1,2,3, 7,8,9, 4,5,6] # xyz acceleration for left arm, right arm, left leg, right leg

        filename = os.path.join(self.data_base_path, filename)

        IMU_data = pd.read_csv(filename, index_col=False).to_numpy() # 250, 14 for wear; 150, 64 for cmu

        IMU_start = int(video_frame_id_list[0] / video_duration * IMU_data.shape[0])
        IMU_end = int(video_frame_id_list[-1] / video_duration * IMU_data.shape[0])
        IMU_data = IMU_data[IMU_start:IMU_end, :] # [~64, 64]

        for imu_idx in imu_to_use:
            # no mixup
            if filename2 == None:
                one_IMU_wave = IMU_data[:, imu_idx]
                # one_IMU_wave to float
                one_IMU_wave = one_IMU_wave.astype(np.float32)
                raw_list.append(one_IMU_wave)
                waveform = torch.from_numpy(one_IMU_wave[:,None]).transpose(0,1)
                waveform = waveform - waveform.mean()
            # mixup
            else:
                waveform1, sr = torchaudio.load(filename)
                waveform2, _ = torchaudio.load(filename2)

                waveform1 = waveform1 - waveform1.mean()
                waveform2 = waveform2 - waveform2.mean()

                if waveform1.shape[1] != waveform2.shape[1]:
                    if waveform1.shape[1] > waveform2.shape[1]:
                        # padding
                        temp_wav = torch.zeros(1, waveform1.shape[1])
                        temp_wav[0, 0:waveform2.shape[1]] = waveform2
                        waveform2 = temp_wav
                    else:
                        # cutting
                        waveform2 = waveform2[0, 0:waveform1.shape[1]]

                mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
                waveform = mix_waveform - mix_waveform.mean()

            try:
                resample_ratio = 320/waveform.shape[1]
                resample_target = int(100*resample_ratio)
                waveform = torchaudio.transforms.Resample(100, resample_target)(waveform)

                spectrogram = self.spectrogram_transform(waveform) # from [1,320] to [1,129,321]
                spectrogram_db = torchaudio.transforms.AmplitudeToDB()(spectrogram)
                fbank = spectrogram_db.squeeze(0).transpose(0,1)[:,:128] # [1,129,321] -> [129,321] -> [321,129] -> [321,128]
            except:
                fbank = torch.zeros([512, 128]) + 0.01
                print('there is a loading error')
                exit()

            target_length = self.target_length
            n_frames = fbank.shape[0]

            p = target_length - n_frames

            # cut and pad
            if p > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, p))
                fbank = m(fbank)
            elif p < 0:
                fbank = fbank[0:target_length, :]

            fbank = fbank.unsqueeze(0)
            fbank_list.append(fbank)
        
        fbank_cat = torch.cat(fbank_list, dim=0) 
        raw_cat = np.array(raw_list) # [12, 60]

        return (fbank_cat, raw_cat)

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length + 1) // self.num_segments # skip_length = 32, num_segments = 1
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),average_duration)
            offsets = offsets + np.random.randint(average_duration,size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(np.random.randint(
                num_frames - self.skip_length + 1,
                size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter: # False
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets

    def _video_TSN_decord_batch_loader(self, directory, video_reader, duration, indices, skip_offsets):
        sampled_list = []
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)): # skip_length = 32, new_step = 2
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step

        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            sampled_list = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in enumerate(frame_id_list)]
        except:
            raise RuntimeError('Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, directory, duration))
        return sampled_list, frame_id_list, duration

    def get_video(self, video_name):
        video_name = os.path.join(self.data_base_path, video_name)
        decord_vr = decord.VideoReader(video_name, num_threads=1)
        duration = len(decord_vr)
        segment_indices, skip_offsets = self._sample_train_indices(duration)
        images, frame_id_list, duration = self._video_TSN_decord_batch_loader(video_name, decord_vr, duration, segment_indices, skip_offsets)

        if self.save_visualization_path != None:
            if not os.path.exists(self.save_visualization_path):
                os.makedirs(self.save_visualization_path)
            # save images as a video with PIL
            save_video_name = '%s/%s.gif' % (self.save_visualization_path, video_name.split('/')[-1].split('.')[0])
            print('save video to', save_video_name)
            images[0].save(save_video_name, save_all=True, append_images=images[1:], duration=200, loop=0)
            
        process_data, mask = self.video_transform((images, None)) # T*C,H,W
        process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W

        if self.image_as_video:
            chosen_image = process_data[:, 8, :, :]
            # fill process_data with chosen_image
            process_data = chosen_image.unsqueeze(1).repeat(1, 16, 1, 1)

        return process_data, mask, frame_id_list, duration

    def __getitem__(self, index):

        if random.random() < self.mixup:
            print('should not reach here not implemented')
            exit()

        else:
            datum = self.data[index]
            datum = self.decode_data(datum)

            process_data, mask, video_frame_id_list, video_duration = self.get_video(datum['video_path'])

            try:
                fbank, raw_imu = self._imu2fbank(datum['imu'], video_frame_id_list, video_duration)
                fbank = fbank.to(torch.float32)
            except:
                fbank = torch.zeros([self.target_length, 6]) + 0.01
                print('there is an error in loading imu')
                exit()

            if self.save_visualization_path != None:
                if not os.path.exists(self.save_visualization_path):
                    os.makedirs(self.save_visualization_path)
                # plot and save raw_imu as an image with PIL
                save_imu_name = '%s/%s_1.png' % (self.save_visualization_path, datum['video_id'])
                print('save imu to', save_imu_name)
                import matplotlib.pyplot as plt
                # raw_imu.shape [12, 60]

                # plot 12 subplots in 3 rows and 4 columns
                # the first 3 are x, y, z acceleration for left arm
                # the first 3 should be in one row
                save_imu_name = '%s/%s_1.png' % (self.save_visualization_path, datum['video_id'])
                print('save imu to', save_imu_name)
                fig, axs = plt.subplots(4, 3, figsize=(20,15))
                for j in range(4):
                    for i in range(3):
                        axs[j, i].plot(raw_imu[i+3*j,:], marker='o')
                plt.savefig(save_imu_name)
                plt.close()
                

                # plot and save fbank as an image with PIL
                save_fbank_name = '%s/%s_2.png' % (self.save_visualization_path, datum['video_id'])
                print('save fbank to', save_fbank_name)
                print(fbank.shape, 'fbank')
                fig, axs = plt.subplots(4, 3, figsize=(20,15))
                for j in range(4):
                    for i in range(3):
                        this_fbank = fbank[i+3*j, ...].numpy().transpose(1,0)
                        axs[j, i].imshow(this_fbank)
                plt.savefig(save_fbank_name)
                plt.close()
                exit()
                
            # label smooth for negative samples, epsilon/label_num
            label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num) 
            if datum['labels'] != '0': # fine-tune
                for label_str in datum['labels'].split(','):
                    label_indices[int(self.index_dict[label_str])] = 1.0 - self.label_smooth
            label_indices = torch.FloatTensor(label_indices)

        # normalize the input for both training and test
        if self.skip_norm == False:
            fbank = (fbank - self.norm_mean) / (self.norm_std)
        # skip normalization the input ONLY when you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            if self.use_imu:
                fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1], fbank.shape[2]) * np.random.rand() / 10
                # fbank = torch.roll(fbank, np.random.randint(-self.target_length, self.target_length), 0)
            else:
                fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
                fbank = torch.roll(fbank, np.random.randint(-self.target_length, self.target_length), 0)

        return fbank, process_data, datum['video_path'], label_indices

    def __len__(self):
        return self.num_samples