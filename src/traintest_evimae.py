# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler

def train(evi_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    batch_time, per_sample_time, data_time, per_sample_data_time, per_sample_dnn_time = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_av_meter, loss_a_meter, loss_v_meter, loss_c_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    loss_g_meter = AverageMeter()
    progress = []

    best_epoch, best_loss = 0, np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_loss,
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if not isinstance(evi_model, nn.DataParallel):
        evi_model = nn.DataParallel(evi_model)

    evi_model = evi_model.to(device)
    trainables = [p for p in evi_model.parameters() if p.requires_grad]
    
    print('Total parameter number is : {:.3f}'.format(sum(p.numel() for p in evi_model.parameters())))
    print('Total trainable parameter number is : {:.3f}'.format(sum(p.numel() for p in trainables)))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    # use adapt learning rate scheduler, for preliminary experiments only, should not use for formal experiments
    if args.lr_adapt == True:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
        print('Override to use adaptive learning rate scheduler.')
    else: # pretrain here
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, args.lrscheduler_step)),gamma=args.lrscheduler_decay)
        print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} every {:d} epoches'.format(args.lrscheduler_start, args.lrscheduler_decay, args.lrscheduler_step))

    print('now training with {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(scheduler)))

    # #optional, save epoch 0 untrained model, for ablation study on model initialization purpose
    # torch.save(evi_model.state_dict(), "%s/models/evi_model.%d.pth" % (exp_dir, epoch))

    epoch += 1
    scaler = GradScaler()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 12])  # for each epoch, 10 metrics to record
    evi_model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        evi_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))
        print('current masking ratio is {:.3f} for imu modality, {:.3f} for video; imu mask mode {:s}'.format(args.imu_masking_ratio, args.video_masking_ratio, args.imu_mask_mode))

        for i, (a_input, v_input, v_mask, _) in enumerate(train_loader):

            # print(a_input.shape, v_input.shape, v_mask.shape)
            # [b, 12, 48, 64], [b, 3, 16, 224, 224], [b, 1568]

            B = a_input.size(0)
            a_input = a_input.to(device, non_blocking=True)
            v_input = v_input.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / a_input.shape[0])
            dnn_start_time = time.time()

            with autocast():
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc, loss_g = evi_model(a_input, v_input, v_mask, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.imu_mask_mode)
                # this is due to for torch.nn.DataParallel, the output loss of 4 gpus won't be automatically averaged, need to be done manually
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc, loss_g = loss.sum(), loss_mae.sum(), loss_mae_a.sum(), loss_mae_v.sum(), loss_c.sum(), c_acc.mean(), loss_g.sum()

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # loss_av is the main loss
            loss_av_meter.update(loss.item(), B)
            loss_a_meter.update(loss_mae_a.item(), B)
            loss_v_meter.update(loss_mae_v.item(), B)
            loss_c_meter.update(loss_c.item(), B)
            loss_g_meter.update(loss_g.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/a_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/a_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Total Loss {loss_av_meter.val:.4f}\t'
                  'Train MAE Loss imu {loss_a_meter.val:.4f}\t'
                  'Train MAE Loss Visual {loss_v_meter.val:.4f}\t'
                  'Train Contrastive Loss {loss_c_meter.val:.4f}\t'
                  'Train Contrastive Acc {c_acc:.3f}\t'
                  'Train Graph Loss {loss_g_meter.val:.4f}\t'
                  .format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_av_meter=loss_av_meter, loss_a_meter=loss_a_meter, loss_v_meter=loss_v_meter, loss_c_meter=loss_c_meter, c_acc=c_acc, loss_g_meter=loss_g_meter), flush=True)
                if np.isnan(loss_av_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        print('start validation')
        eval_loss_av, eval_loss_mae, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_c_acc, eval_loss_g = validate(evi_model, test_loader, args)

        print("Eval imu MAE Loss: {:.6f}".format(eval_loss_mae_a))
        print("Eval Visual MAE Loss: {:.6f}".format(eval_loss_mae_v))
        print("Eval Total MAE Loss: {:.6f}".format(eval_loss_mae))
        print("Eval Contrastive Loss: {:.6f}".format(eval_loss_c))
        print("Eval Total Loss: {:.6f}".format(eval_loss_av))
        print("Eval Contrastive Accuracy: {:.6f}".format(eval_c_acc))
        print("Eval Graph Loss: {:.6f}".format(eval_loss_g))

        print("Train imu MAE Loss: {:.6f}".format(loss_a_meter.avg))
        print("Train Visual MAE Loss: {:.6f}".format(loss_v_meter.avg))
        print("Train Contrastive Loss: {:.6f}".format(loss_c_meter.avg))
        print("Train Total Loss: {:.6f}".format(loss_av_meter.avg))
        print("Train Graph Loss: {:.6f}".format(loss_g_meter.avg))

        # train imu mae loss, train visual mae loss, train contrastive loss, train total loss
        # eval imu mae loss, eval visual mae loss, eval contrastive loss, eval total loss, eval contrastive accuracy, lr
        result[epoch-1, :] = [loss_a_meter.avg, loss_v_meter.avg, loss_c_meter.avg, loss_av_meter.avg, eval_loss_mae_a, eval_loss_mae_v, eval_loss_c, eval_loss_av, eval_c_acc, optimizer.param_groups[0]['lr'], loss_g_meter.avg, eval_loss_g]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')

        if eval_loss_av < best_loss:
            best_loss = eval_loss_av
            best_epoch = epoch

        if best_epoch == epoch:
            torch.save(evi_model.state_dict(), "%s/models/best_evi_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))

        if args.save_model == True:
            if epoch in [50, 100, 150, 200, 250, 299]:
                torch.save(evi_model.state_dict(), "%s/models/evi_model.%d.pth" % (exp_dir, epoch))


        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(-eval_loss_av)
        else:
            scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        per_sample_dnn_time.reset()
        loss_av_meter.reset()
        loss_a_meter.reset()
        loss_v_meter.reset()
        loss_c_meter.reset()
        loss_g_meter.reset()

def validate(evi_model, val_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(evi_model, nn.DataParallel):
        evi_model = nn.DataParallel(evi_model)
    evi_model = evi_model.to(device)
    evi_model.eval()

    end = time.time()
    A_loss, A_loss_mae, A_loss_mae_a, A_loss_mae_v, A_loss_c, A_c_acc, A_loss_g = [], [], [], [], [], [], []
    with torch.no_grad():
        for i, (a_input, v_input, v_mask, _) in enumerate(val_loader):
            a_input = a_input.to(device)
            v_input = v_input.to(device)
            with autocast():
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, mask_a, mask_v, c_acc, loss_g = evi_model(a_input, v_input, v_mask, mae_loss_weight=args.mae_loss_weight, contrast_loss_weight=args.contrast_loss_weight, mask_mode=args.imu_mask_mode)
                loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc, loss_g = loss.sum(), loss_mae.sum(), loss_mae_a.sum(), loss_mae_v.sum(), loss_c.sum(), c_acc.mean(), loss_g.sum()
            A_loss.append(loss.to('cpu').detach())
            A_loss_mae.append(loss_mae.to('cpu').detach())
            A_loss_mae_a.append(loss_mae_a.to('cpu').detach())
            A_loss_mae_v.append(loss_mae_v.to('cpu').detach())
            A_loss_c.append(loss_c.to('cpu').detach())
            A_c_acc.append(c_acc.to('cpu').detach())
            A_loss_g.append(loss_g.to('cpu').detach())
            batch_time.update(time.time() - end)
            end = time.time()

        loss = np.mean(A_loss)
        loss_mae = np.mean(A_loss_mae)
        loss_mae_a = np.mean(A_loss_mae_a)
        loss_mae_v = np.mean(A_loss_mae_v)
        loss_c = np.mean(A_loss_c)
        c_acc = np.mean(A_c_acc)
        loss_g = np.mean(A_loss_g)

    return loss, loss_mae, loss_mae_a, loss_mae_v, loss_c, c_acc, loss_g