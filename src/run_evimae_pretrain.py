# -*- coding: utf-8 -*-
# @Time    : 6/11/21 12:57 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : run.py

import argparse
import os
import ast
import pickle
import sys
import time
import json
import torch
from torch.utils.data import WeightedRandomSampler
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloader as dataloader
import models
import numpy as np
from traintest_evimae import train

# pretrain evi-mae model

print("I am process %s, running on %s: starting (%s)" % (os.getpid(), os.uname()[1], time.asctime()))

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# dataset
parser.add_argument("--dataset", type=str, default="cmummac", help="the dataset used", choices=["wear","cmummac"])
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label-csv", type=str, default='', help="csv with class labels")
parser.add_argument("--n_class", type=int, default=0, help="number of classes")

# training
parser.add_argument("--model", type=str, default='evi-mae', help="the model used")
parser.add_argument("--noise", help='if use balance sampling', default='True', type=ast.literal_eval)
parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=12, type=int, metavar='N', help='mini-batch size')
parser.add_argument('-w', '--num-workers', default=16, type=int, metavar='NW', help='# of workers for dataloading (default: 32)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument("--lr_adapt", help='if use adaptive learning rate', default='False', type=ast.literal_eval)
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics in finetuning", choices=["mAP", "acc"])
parser.add_argument('--warmup', help='if use warmup learning rate scheduler', type=ast.literal_eval, default='True')
parser.add_argument("--lrscheduler_start", default=10, type=int, help="when to start decay in finetuning")
parser.add_argument("--lrscheduler_step", default=5, type=int, help="the number of step to decrease the learning rate in finetuning")
parser.add_argument("--lrscheduler_decay", default=0.5, type=float, help="the learning rate decay ratio in finetuning")
parser.add_argument("--n-print-steps", type=int, default=100, help="number of steps to print statistics")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--bal", type=str, default=None, help="use balanced sampling or not")
parser.add_argument("--cont_model", help='previous pretrained model', type=str, default=None)
parser.add_argument("--weight_file", type=str, default=None, help="path to weight file")
parser.add_argument('--norm_pix_loss', help='if use norm_pix_loss', type=ast.literal_eval, default=None)
parser.add_argument("--pretrain_path", type=str, default='None', help="pretrained model path")
parser.add_argument("--contrast_loss_weight", type=float, default=0.01, help="weight for contrastive loss")
parser.add_argument("--mae_loss_weight", type=float, default=3.0, help="weight for mae loss")
parser.add_argument('--tr_pos', help='if use trainable positional embedding', type=ast.literal_eval, default='False')
parser.add_argument("--load_prepretrain", help='if load pre-pretrain', type=ast.literal_eval, default='True')
parser.add_argument("--pretrain_modality", type=str, default='both', help="pretrain modality", choices=['imu', 'video', 'both'])
parser.add_argument("--image_as_video", help='if use image as video', type=ast.literal_eval, default='False')

# imu
parser.add_argument("--imu_target_length", type=int, default=48, help="the target length of imu data")
parser.add_argument("--imu_masking_ratio", type=float, default=0.75, help="masking ratio")
parser.add_argument("--imu_mask_mode", type=str, default='unstructured', help="masking ratio", choices=['unstructured', 'time', 'freq', 'tf'])
parser.add_argument("--imu_plot_type", type=str, default='fbank', help="the plot type of imu data", choices=['fbank', 'rp', 'mel', 'raw', 'stft']) 
parser.add_argument("--imu_plot_height", type=int, default=64, help="the plot height of imu data")
parser.add_argument("--imu_patch_size", type=int, default=8, help="the patch size of imu data")
parser.add_argument("--imu_dataset_mean", type=float, help="the dataset imu mean, used for input normalization")
parser.add_argument("--imu_dataset_std", type=float, help="the dataset imu std, used for input normalization")
parser.add_argument("--imu_channel_num", type=int, default=12, help="the channel number of imu data")
parser.add_argument("--imu_encoder_embed_dim", type=int, default=384, help="the embed dim of imu encoder")
parser.add_argument("--imu_encoder_depth", type=int, default=12, help="the depth of imu encoder")
parser.add_argument("--imu_encoder_num_heads", type=int, default=6, help="the num heads of imu encoder")
parser.add_argument("--imu_enable_graph", type=ast.literal_eval, default='False', help="enable graph for imu data")
parser.add_argument("--imu_graph_net", type=str, default='gin', help="the graph net for imu data", choices=['gin']) # dont use gat
parser.add_argument("--imu_graph_masking_ratio", type=float, default=0.5, help="masking ratio for graph net")

# video
parser.add_argument("--video_img_size", type=int, default=224, help="the image size of video data")
parser.add_argument("--video_patch_size", type=int, default=16, help="the patch size of video data")
parser.add_argument("--video_encoder_num_classes", type=int, default=0, help="the number of classes of video encoder")
parser.add_argument("--video_decoder_num_classes", type=int, default=1536, help="the number of classes of video decoder")
parser.add_argument("--video_mlp_ratio", type=int, default=4, help="the mlp ratio of video data")
parser.add_argument("--video_qkv_bias", type=ast.literal_eval, default='True', help="the qkv bias of video data")
parser.add_argument("--video_encoder_embed_dim", type=int, default=384, help="the embed dim of video encoder")
parser.add_argument("--video_encoder_depth", type=int, default=12, help="the depth of video encoder")
parser.add_argument("--video_encoder_num_heads", type=int, default=6, help="the num heads of video encoder")
parser.add_argument("--video_decoder_embed_dim", type=int, default=192, help="the embed dim of video decoder")
parser.add_argument("--video_decoder_num_heads", type=int, default=3, help="the num heads of video decoder")
parser.add_argument("--video_masking_ratio", type=float, default=0.75, help="video masking ratio")

args = parser.parse_args()

im_res = args.video_img_size
imu_conf = {'num_mel_bins': 128, 'target_length': args.imu_target_length, 'freqm': 0, 'timem': 0, 'mixup': args.mixup, 'dataset': args.dataset, 'mode':'train', 'mean':args.imu_dataset_mean, 'std':args.imu_dataset_std,
              'noise':args.noise, 'label_smooth': 0, 'im_res': im_res}
val_imu_conf = {'num_mel_bins': 128, 'target_length': args.imu_target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset,
                  'mode':'eval', 'mean': args.imu_dataset_mean, 'std': args.imu_dataset_std, 'noise': False, 'im_res': im_res}

print('current mae loss {:.3f}, and contrastive loss {:.3f}'.format(args.mae_loss_weight, args.contrast_loss_weight))



if args.bal == 'bal':
    print('balanced sampler is being used')
    if args.weight_file == None:
        samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
    else:
        samples_weight = np.loadtxt(args.data_train[:-5] + '_' + args.weight_file + '.csv', delimiter=',')
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        dataloader.EVIDataset(args.data_train, label_csv=args.label_csv, imu_conf=imu_conf),
        batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True, drop_last=True)
else:
    print('balanced sampler is not used')
    train_loader = torch.utils.data.DataLoader(
        dataloader.EVIDataset(args.data_train, label_csv=args.label_csv, imu_conf=imu_conf, video_masking_ratio=args.video_masking_ratio, image_as_video=args.image_as_video),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

val_loader = torch.utils.data.DataLoader(
    dataloader.EVIDataset(args.data_val, label_csv=args.label_csv, imu_conf=val_imu_conf),
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

if args.data_eval != None:
    eval_loader = torch.utils.data.DataLoader(
        dataloader.EVIDataset(args.data_eval, label_csv=args.label_csv, imu_conf=val_imu_conf),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)

if args.model == 'evi-mae':
    print('pretrain a evi-mae model with x modality-specific layers and 1 modality-sharing layers')
    video_model_dict = {
        'img_size': args.video_img_size,
        'patch_size': args.video_patch_size,
        'encoder_embed_dim': args.video_encoder_embed_dim,
        'encoder_depth': args.video_encoder_depth,
        'encoder_num_heads': args.video_encoder_num_heads,
        'mlp_ratio': args.video_mlp_ratio,
        'qkv_bias': args.video_qkv_bias,
        'encoder_num_classes': args.video_encoder_num_classes,
        'decode_num_classes': args.video_decoder_num_classes,
        'decode_embed_dim': args.video_decoder_embed_dim,
        'decode_num_heads': args.video_decoder_num_heads,
        'masking_ratio': args.video_masking_ratio,
        'pretrain_modality': args.pretrain_modality,
    }

    imu_model_dict = {
        'target_length': args.imu_target_length,
        'masking_ratio': args.imu_masking_ratio,
        'mask_mode': args.imu_mask_mode,
        'plot_type': args.imu_plot_type,
        'plot_height': args.imu_plot_height,
        'patch_size': args.imu_patch_size,
        'channel_num': args.imu_channel_num,
        'encoder_embed_dim': args.imu_encoder_embed_dim,
        'encoder_depth': args.imu_encoder_depth,
        'encoder_num_heads': args.imu_encoder_num_heads,
        'enable_graph': args.imu_enable_graph,
        'imu_graph_net': args.imu_graph_net,
        'imu_graph_masking_ratio': args.imu_graph_masking_ratio,
    }
    evi_model = models.EVIMAE(norm_pix_loss=args.norm_pix_loss, tr_pos=args.tr_pos, video_model_dict=video_model_dict, imu_model_dict=imu_model_dict)
else:
    raise ValueError('model not supported')

# initialized with a pretrained checkpoint (adapted from original video-MAE checkpoint)
# for the IMU branch, we initialize it with the ImageMAE checkpoint, but experiments show that it is not necessary
if args.pretrain_path != 'None' and args.load_prepretrain:
    if os.path.isfile(args.pretrain_path): pass
    else: raise ValueError('pretrained model not found')

    mdl_weight = torch.load(args.pretrain_path, map_location=torch.device('cpu'))
    if not isinstance(evi_model, torch.nn.DataParallel):
        evi_model = torch.nn.DataParallel(evi_model)

    useful_weight = {}
    for key in mdl_weight.keys():
        if 'graph' not in key:
            useful_weight[key] = mdl_weight[key]

    miss, unexpected = evi_model.load_state_dict(useful_weight, strict=False)
    print('now load mae pretrained weights from ', args.pretrain_path)
    print('miss', miss)
    print('unexpected', unexpected)

print("Creating experiment directory: %s" % args.exp_dir)
try:
    os.makedirs(os.path.join(args.exp_dir, "models"), exist_ok=True)
except:
    pass
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)
with open(args.exp_dir + '/args.json', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

print('Now starting training for {:d} epochs.'.format(args.n_epochs))
train(evi_model, train_loader, val_loader, args)