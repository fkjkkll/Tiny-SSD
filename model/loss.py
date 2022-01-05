""" ヾ(•ω•`)o coding:utf-8
Auther: lee
Date: 2021/12/28
Time: 14:45:24
"""

from torch.nn import functional as F
import torch


def focal_loss(cls_preds, cls_labels, gamma=2):
    """ 计算类别损失
    :param cls_preds: (bs, anchors, 1+c)
    :param cls_labels: (bs, anchors)
    :param gamma: int default->2
    :return: (bs,)
    """
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls_preds = cls_preds.reshape(-1, num_classes)
    cls_labels = cls_labels.reshape(-1)
    cls_preds = F.softmax(cls_preds, dim=1)
    x = cls_preds[torch.arange(cls_labels.shape[0]), cls_labels]
    cls = -(1 - x) ** gamma * torch.log(x)
    return cls.reshape(batch_size, -1).mean(dim=1)


def smooth_l1_loss(bbox_preds, bbox_labels, bbox_masks, sigma=1):
    """ 计算类边框损失
    :param bbox_preds: (bs, anchors*4)
    :param bbox_labels: (bs, anchors*4)
    :param bbox_masks: (bs, anchors*4)
    :param sigma: int default->1
    :return: (bs,)
    """
    bbox_preds = bbox_preds * bbox_masks
    bbox_labels = bbox_labels * bbox_masks
    active_items = bbox_masks.sum(axis=1) + 1e-5
    beta = 1. / (sigma ** 2)
    diff = torch.abs(bbox_preds - bbox_labels)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return torch.sum(loss, dim=1) / active_items


def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks, alpha = 0.01):
    """ 总体损失函数
    :param cls_preds: (bs, anchors, 1+c)
    :param cls_labels: (bs, anchors)
    :param bbox_preds: (bs, anchors*4)
    :param bbox_labels: (bs, anchors*4)
    :param bbox_masks: (bs, anchors*4)
    :param alpha: 平衡类别损失与边框损失的系数
    :return: int
    """
    cls_loss = focal_loss(cls_preds, cls_labels)
    bbox_loss = smooth_l1_loss(bbox_preds, bbox_labels, bbox_masks)
    return (cls_loss + alpha * bbox_loss).mean()


@torch.no_grad()
def cls_eval(cls_preds, cls_labels):
    """ 类别分类损失评估
    :param cls_preds: (bs, anchors, 1+c)
    :param cls_labels: (bs, anchors)
    :return: int
    """
    return focal_loss(cls_preds, cls_labels).mean()


@torch.no_grad()
def bbox_eval(bbox_preds, bbox_labels, bbox_masks, alpha = 0.01):
    """ 边框回归损失评估
    :param bbox_preds: (bs, anchors*4)
    :param bbox_labels: (bs, anchors*4)
    :param bbox_masks: (bs, anchors*4)
    :param alpha: 平衡类别损失与边框损失的系数
    :return: int
    """
    return alpha * smooth_l1_loss(bbox_preds, bbox_labels, bbox_masks).mean()