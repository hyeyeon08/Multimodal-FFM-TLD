# coding:utf-8
import os
import argparse
import numpy as np
import glob
import time
import copy
import pandas as pd
import torch
import torch_optimizer

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from collections import defaultdict
from tqdm import tqdm
from util.PL_dataset import PL_dataset
from util.util import calc_loss, print_metrics

from model import Unet_original_4c, Unet_proposed

# config
n_class   = 2
base_dir  = '__'
vl_dir    = '__'
ir_dir    = '__'
gt_dir    = '__'
model_dir = '__'
data_dir  = '__'
lr_start  = 0.001

def train(model, train_loader, scheduler, optimizer, epo):

    start_t = t = time.time()
    scheduler.step()
    for param_group in optimizer.param_groups:
        print("LR", param_group['lr'])

    model.train()  # Set model to training mode
    metrics = defaultdict(float)

    for it, (input_vl, input_ir, labels) in enumerate(train_loader):
        input_vl = Variable(input_vl).cuda(args.gpu)
        input_ir = Variable(input_ir).cuda(args.gpu)
        labels = Variable(labels).cuda(args.gpu)
        if args.gpu >= 0:
            input_vl = input_vl.cuda(args.gpu)
            input_ir = input_ir.cuda(args.gpu)
            labels = labels.cuda(args.gpu)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward. track history if only in train
        outputs = model(input_vl, input_ir)
        loss, metrics = calc_loss(outputs, labels, metrics)

        # backward + optimize only if in training phase
        loss.backward()
        optimizer.step()

        cur_t = time.time()
        if cur_t - t > 5:
            print('|- epo %s/%s. train iter %s/%s. %.2f img/sec loss: %.4f' \
                  % (epo, args.epoch_max, it + 1, train_loader.n_iter, (it + 1) * args.batch_size/(cur_t - start_t),
                  float(loss)))
            t += 5

    t_bce, t_dice, t_loss = metrics['bce']/(it + 1), metrics['dice']/(it + 1), metrics['loss'] / (it + 1)
    content = '| epo:%s/%s train_loss_avg:%.4f bce:%.4f dice:%.4f' \
              % (epo, args.epoch_max, t_loss, t_bce, t_dice)
    print(content)
    with open(log_file, 'a') as appender:
        appender.write(content)

    return t_bce, t_dice, t_loss


def validation(model, val_loader, epo):

    start_t = t = time.time()
    metrics = defaultdict(float)

    with torch.no_grad():
        for it, (input_vl, input_ir, labels) in enumerate(val_loader):
            input_vl = Variable(input_vl)
            input_ir = Variable(input_ir)
            labels = Variable(labels)
            if args.gpu >= 0:
                input_vl = input_vl.cuda(args.gpu)
                input_ir = input_ir.cuda(args.gpu)
                labels = labels.cuda(args.gpu)

            outputs = model(input_vl, input_ir)
            loss, metrics = calc_loss(outputs, labels, metrics)

            cur_t = time.time()
            if cur_t - t > 5:
                print('|- epo %s/%s. val iter %s/%s. %.2f img/sec loss: %.4f' \
                    % (epo, args.epoch_max, it+1, val_loader.n_iter, (it+1)*args.batch_size/(cur_t-start_t), float(loss)))
                t += 5

    v_bce, v_dice, v_loss = metrics['bce'] / (it + 1), metrics['dice'] / (it + 1), metrics['loss'] / (it + 1)
    epoch_loss = metrics['loss']/(it+1)
    content = '| epo:%s/%s val_loss_avg:%.4f bce:%.4f dice:%.4f' \
              % (epo, args.epoch_max, v_loss, v_bce, v_dice)
    print(content)
    with open(log_file, 'a') as appender:
        appender.write(content)

    return v_bce, v_dice, v_loss, epoch_loss


def main():

    model = eval(args.model_name)(n_class=n_class)
    if args.gpu >= 0: model.cuda(args.gpu)
    optimizer = torch_optimizer.RAdam(
    filter(lambda p: p.requires_grad, model.parameters()),
        lr= lr_start,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0005,
    )

    if args.epoch_from > 1:
        print('| loading checkpoint file %s... ' % checkpoint_model_file, end='')
        model.load_state_dict(torch.load(checkpoint_model_file))
        optimizer.load_state_dict(torch.load(checkpoint_optim_file))
        print('done!')

    vl_tra_list = "__load your data__"
    ir_tra_list = "__load your data__"
    gt_tra_list = "__load your data__"
    vl_val_list = "__load your data__"
    ir_val_list = "__load your data__"
    gt_val_list = "__load your data__"

    print('Training data set', len(vl_tra_list), len(ir_tra_list), len(gt_tra_list))
    print('Validation data set', len(vl_val_list), len(ir_val_list), len(gt_val_list))
    train_dataset = PL_dataset(vl_tra_list, ir_tra_list, gt_tra_list, is_train=True)
    val_dataset  = PL_dataset(vl_val_list, ir_val_list, gt_val_list, is_train=False)

    train_loader  = DataLoader(
        dataset     = train_dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = True
    )
    val_loader  = DataLoader(
        dataset     = val_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    train_loader.n_iter = len(train_loader)
    val_loader.n_iter   = len(val_loader)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10

    loss_train_save = []
    loss_val_save = []

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    for epo in tqdm(range(args.epoch_from, args.epoch_max+1)):
        print('\n| epo #%s begin...' % epo)

        t_bce, t_dice, t_loss = train(model, train_loader, exp_lr_scheduler, optimizer, epo)
        loss_train_save.append([t_bce, t_dice, t_loss])

        v_bce, v_dice, v_loss, epoch_loss = validation(model, val_loader, epo)
        loss_val_save.append([v_bce, v_dice, v_loss])

        if epoch_loss < best_loss:
            print('| best model file... ', end='')
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print('| saving check point model file... ', end='')
        torch.save(model.state_dict(), checkpoint_model_file)
        torch.save(optimizer.state_dict(), checkpoint_optim_file)

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), final_model_file)
    print('| Training done! \n | Best validation loss: %.4f'\
          % best_loss)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Num of model parameters:", params)

    df1 = pd.DataFrame(loss_train_save, columns=['bce', 'dice', 'loss'])
    df2 = pd.DataFrame(loss_val_save, columns=['bce', 'dice', 'loss'])
    with pd.ExcelWriter(save_dir + '/'+ args.model_name+ '_'
                                   'log_'+'seed' + str(args.rand_seed)+'.xlsx') as writer:
        df1.to_excel(writer, sheet_name = 'train')
        df2.to_excel(writer, sheet_name = 'val')

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Train FFM with pytorch')
    parser.add_argument('--model_name',  '-M',  type=str, default='UNet_proposed')#
    parser.add_argument('--batch_size',  '-B',  type=int, default=5)
    parser.add_argument('--epoch_max' ,  '-E',  type=int, default=200)
    parser.add_argument('--epoch_from',  '-EF', type=int, default=1)
    parser.add_argument('--gpu',         '-G',  type=int, default=0)
    parser.add_argument('--num_workers', '-j',  type=int, default=0)
    parser.add_argument('--rand_seed',   '-RS', type=int, default=1)
    args = parser.parse_args()


    save_dir = os.path.join(model_dir, args.model_name)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_model_file = os.path.join(save_dir, args.model_name + '.pth')
    checkpoint_optim_file = os.path.join(save_dir, args.model_name + '.optim')
    final_model_file      = os.path.join(save_dir, args.model_name + '_final_'+'seed' + str(args.rand_seed)+'.pth')
    log_file              = os.path.join(save_dir, args.model_name + '_log_'+'seed' + str(args.rand_seed)+'.txt')

    print('| training %s on GPU #%d with pytorch' % (args.model_name, args.gpu))
    print('| from epoch %d / %s' % (args.epoch_from, args.epoch_max))
    print('| model will be saved in: %s' % save_dir)

    main()
