# coding:utf-8
import torch.nn.functional as F
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()

def calc_loss (pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy()
    metrics['dice'] += dice.data.cpu().numpy()
    metrics['loss'] += loss.data.cpu().numpy()
    return loss, metrics

def calc_loss_aux (pred, auxiliary, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    auxiliary_bce = nn.BCEWithLogitsLoss()
    auxiliary_loss = auxiliary_bce(auxiliary, target[:, 1, :, :].unsqueeze(1))

    loss = bce * bce_weight + dice * (1 - bce_weight) + 0.5*auxiliary_loss

    metrics['bce'] += bce.data.cpu().numpy()
    metrics['dice'] += dice.data.cpu().numpy()
    metrics['aux'] += auxiliary_loss.data.cpu().numpy()
    metrics['loss'] += loss.data.cpu().numpy()
    return loss, metrics


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))

def line_level_estimator(pred, labels, batch_size, iou_threshold=0.5):

    img_recall_tp = []
    img_recall_gt = []
    pred = pred.data.cpu().numpy()
    labels = labels.data.cpu().numpy()

    for batch_idx in range(batch_size):
        pred_ = ((pred[batch_idx, 1, :, :] > 0.5) * 255).astype('uint8')
        label_ = (labels[batch_idx, 1, :, :] * 255).astype('uint8')

        cnts_gt, _ = cv2.findContours(label_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        gt = 0  # the number of gt lines
        tp = 0
        for blob in cnts_gt:
            # 9 is minimum number of pixels to assume a power line
            if len(blob) > 9:
                gt += 1
                maxIOU = 0.01

                # img1: gt image containing a power line  / img2 = intersection of gt and prediction
                blank = np.zeros(pred_.shape[0:2])
                img1 = cv2.drawContours(blank.copy(), [blob], -1, 1, thickness=-1)
                img2 = img1 * pred_

                intersection = img2 == 255
                union = img1 > 0

                if maxIOU < intersection.sum() / union.sum():
                    maxIOU = intersection.sum() / union.sum()

                if maxIOU > iou_threshold:
                    tp += 1
                    continue

        img_recall_tp.append(tp)
        img_recall_gt.append(gt)

    return img_recall_tp, img_recall_gt


def iou_estimator(pred, labels, batch_size, img_merge=False, iou_threshold=0.5):

    pred = pred.data.cpu().numpy()
    labels = labels.data.cpu().numpy()

    if img_merge:
        # Calculate IoU score after reconstruction into 512 x 512 image
        com_pred = com_lable =np.zeros([512, 512])
        com_pred[:256, :256] = pred[0, 1, :, :]
        com_pred[:256, 256:] = pred[1, 1, :, :]
        com_pred[256:, :256] = pred[2, 1, :, :]
        com_pred[256:, 256:] = pred[3, 1, :, :]
        com_lable[:256, :256] = labels[0, 1, :, :]
        com_lable[:256, 256:] = labels[1, 1, :, :]
        com_lable[256:, :256] = labels[2, 1, :, :]
        com_lable[256:, 256:] = labels[3, 1, :, :]

        pred_ = ((com_pred > iou_threshold) * 255).astype('uint8')
        label_ = (com_lable * 255).astype('uint8')

        gt = (label_ > 0).astype('int32')
        seg = (pred_ > 0).astype('int32')

        iou = np.sum(seg[gt == 1]) / (np.sum(seg) + np.sum(gt) - np.sum(seg[gt == 1]))
        return iou

    else:

        iou_res = []
        for _ in range(batch_size):
            pred_ = ((pred > iou_threshold) * 255).astype('uint8')
            label_ = (labels * 255).astype('uint8')

            gt = (label_ > 0).astype('int32')
            seg = (pred_ > 0).astype('int32')

            iou = np.sum(seg[gt == 1]) / (np.sum(seg) + np.sum(gt) - np.sum(seg[gt == 1]))

            iou_res.append(iou)

    return iou_res


def reverse_transform(inp):
    inp = inp.transpose((1, 2, 0))
    inp = (inp * 255).astype(np.uint8)

    return inp



def calculate_cf(pred, labels, iou_threshold=0.5, n_class=2):
    pred = pred.data.cpu().numpy()
    labels = labels.data.cpu().numpy()

    com_pred = np.zeros([512, 512])
    com_lable = np.zeros([512, 512])
    com_pred[:256, :256] = pred[0, 1, :, :]
    com_pred[:256, 256:] = pred[1, 1, :, :]
    com_pred[256:, :256] = pred[2, 1, :, :]
    com_pred[256:, 256:] = pred[3, 1, :, :]
    com_lable[:256, :256] = labels[0, 1, :, :]
    com_lable[:256, 256:] = labels[1, 1, :, :]
    com_lable[256:, :256] = labels[2, 1, :, :]
    com_lable[256:, 256:] = labels[3, 1, :, :]

    pred_ = ((com_pred > iou_threshold) * 255).astype('uint8')
    label_ = (com_lable * 255).astype('uint8')
    seg = (pred_ > 0).astype('int32')
    gt = (label_ > 0).astype('int32')


    cf = np.zeros((n_class, n_class))
    for gtcid in range(n_class):
        for pcid in range(n_class):
            gt_mask = (gt == gtcid).astype('int32')
            pred_mask = (seg == pcid).astype('int32')
            intersection = gt_mask * pred_mask

            cf[gtcid, pcid] = int(intersection.sum())
    return cf

def calculate_result(cf, n_class=2):


    for cid in range(1, n_class):
        if cf[:,cid].sum() > 0:
            iou = cf[1,1] / (cf[0,1]+cf[1,1]+cf[1,0])
            acc = (cf[0,0] + cf[1,1])/cf.sum()
            f1 = 2*cf[1,1]/(2*cf[1,1]+cf[0,1]+cf[1,0])
            recall = cf[1,1]/(cf[1,0]+cf[1,1])
            precision = cf[1,1]/(cf[0,1]+cf[1,1])

    return iou, acc, f1, recall, precision


