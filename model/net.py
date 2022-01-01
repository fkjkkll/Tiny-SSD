""" ヾ(•ω•`)o coding:utf-8
Auther: lee
Date: 2021/12/28
Time: 14:43:40
"""
from torch import nn
import torch

def gen_cls_predictor(num_inputs, num_anchors, num_classes):
    """ 用于形成类别预测的分支
    :param num_inputs: 输入通道数
    :param num_anchors: 每像素锚框数(anchors per pixel, app)
    :param num_classes: 类别数(c)
    :return: nn.Conv
    """
    return nn.Conv2d(num_inputs, num_anchors * (num_classes + 1), kernel_size=3, padding=1) # 加入背景类


def gen_bbox_predictor(num_inputs, num_anchors):
    """ 用于形成锚框坐标偏移量预测的分支
    :param num_inputs: 输入通道数
    :param num_anchors: 每像素锚框数(anchors per pixel, app)
    :return: nn.Conv
    """
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)


def concat_preds(preds):
    """ 使预测输出按照像素点、锚框连续化，之所以把通道维放到最后是，使得每个Fmap像素预测展开后是一个连续值，方便后续处理
    :param preds: 对于类别预测: list((bs, bpp*(1+c), fhi, fwi)*5. 对于边界框预测: list((bs, app*4, fhi, fwi))*5
    :return: 对于类别预测(bs, anchors*(c+1)). 对于边界框预测(bs, anchors*4)
    """
    def flatten_pred(pred):
        return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1) # (bs, -1)
    return torch.cat([flatten_pred(p) for p in preds], dim=1)


def down_sample_blk(in_channels, out_channels):
    """ 下采样块 (Conv+BN+Relu)*N + Maxpooling
    :param in_channels: 输入通道数
    :param out_channels: 输出通道数
    :return: nn.Sequential
    """
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(2))
    return nn.Sequential(*blk)


def base_net():
    """ 特征提取网络backbone. 由三个下采样块组成
    :return: nn.Sequential
    """
    blk = []
    num_filters = [3, 16, 24, 48]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)


def get_blk(i):
    """ 为了构造网络代码方便
    :param i: index
    :return:
    """
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(48, 64)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1,1))
    else:
        blk = down_sample_blk(64, 64)
    return blk


def blk_forward(X, blk, cls_predictor, bbox_predictor):
    """ stage之间产生分支输出
    :param X: 输入tensor
    :param blk: 各个网络块blk
    :param cls_predictor: Conv, 用于产生该blk输出的类别输出
    :param bbox_predictor: Conv, 用于产生该blk输出的边框偏移输出
    :return: 该层的(主干输出、类别分支、边界框偏移分支)
    """
    Y = blk(X)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return Y, cls_preds, bbox_preds


class TinySSD(nn.Module):
    def __init__(self, anchors_perpixel, num_classes):
        """ 总体网络
        :param anchors_perpixel: 每个像素点分配的锚框数
        :param num_classes: 类别总数(不包含背景)
        """
        super(TinySSD, self).__init__()
        self.num_classes = num_classes
        idx_to_in_channels = [48, 64, 64, 64, 64]
        for i in range(5):
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', gen_cls_predictor(idx_to_in_channels[i], anchors_perpixel, num_classes))
            setattr(self, f'bbox_{i}', gen_bbox_predictor(idx_to_in_channels[i], anchors_perpixel))

    def forward(self, X):
        """ 神经网络前向传播
        :param X: (bs, 3, w, h)
        :return: (bs, anchors, 1+c), (bs, anchors*4)
        """
        cls_preds, bbox_preds = [None] * 5, [None] * 5
        for i in range(5):
            # `getattr(self, 'blk_%d' % i)` 即访问 `self.blk_i`
            X, cls_preds[i], bbox_preds[i] = blk_forward(
                X,
                getattr(self, f'blk_{i}'),
                getattr(self, f'cls_{i}'),
                getattr(self, f'bbox_{i}')
            )
        cls_preds = concat_preds(cls_preds) # (8, 10888)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1) # (8, 5444, 2)
        bbox_preds = concat_preds(bbox_preds) # (8, 21776)
        return cls_preds, bbox_preds