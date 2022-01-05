""" ヾ(•ω•`)o coding:utf-8
Auther: lee
Date: 2021/12/28
Time: 14:43:40
"""
import torch
from torch import nn

# ----------------------------------------------------------------
#                        引出分支用于预测
# ----------------------------------------------------------------
def gen_cls_predictor(in_channels, app, cn):
    """ 用于形成类别预测的分支
    :param in_channels: 输入通道数
    :param app: 每像素锚框数(anchors per pixel, app)
    :param cn: 类别数(class number)
    :return: nn.Conv
    """
    blk = []
    out_channels = app * (cn + 1) # 加入背景类
    # blk.extend([nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
    #             nn.Conv2d(in_channels, out_channels, 1, 1, 0),
    #             ]) # dw pw
    blk.extend([nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                ])
    return nn.Sequential(*blk)


def gen_bbox_predictor(in_channels, app):
    """ 用于形成锚框坐标偏移量预测的分支
    :param in_channels: 输入通道数
    :param app: 每像素锚框数(anchors per pixel, app)
    :return: nn.Conv
    """
    blk = []
    out_channels = app * 4
    # blk.extend([nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
    #             nn.Conv2d(in_channels, out_channels, 1, 1, 0),
    #             ])
    blk.extend([nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                ]) # dw pw
    return nn.Sequential(*blk)


def concat_preds(preds):
    """ 使预测输出按照像素点、锚框连续化，之所以把通道维放到最后是，使得每个Fmap像素预测展开后是一个连续值，方便后续处理
    :param preds: 对于类别预测: list((bs, bpp*(1+c), fhi, fwi)*5. 对于边界框预测: list((bs, app*4, fhi, fwi))*5
    :return: 对于类别预测(bs, anchors*(c+1)). 对于边界框预测(bs, anchors*4)
    """
    def flatten_pred(pred):
        return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1) # (bs, -1)
    return torch.cat([flatten_pred(p) for p in preds], dim=1)


# ----------------------------------------------------------------
#                     搭建主干网络(简单的卷积网络)
# ----------------------------------------------------------------
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
    def __init__(self, app, cn):
        """ 总体网络
        :param app: 每个像素点分配的锚框数
        :param cn: 类别总数(不包含背景)
        """
        super(TinySSD, self).__init__()
        self.num_classes = cn
        idx_to_in_channels = [48, 64, 64, 64, 64]
        for i in range(5):
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', gen_cls_predictor(idx_to_in_channels[i], app, cn))
            setattr(self, f'bbox_{i}', gen_bbox_predictor(idx_to_in_channels[i], app))

    def forward(self, X):
        """ 神经网络前向传播
        :param X: (bs, 3, w, h)
        :return: (bs, anchors, 1+c), (bs, anchors*4)
        """
        cls_preds, bbox_preds = [None] * 5, [None] * 5
        for i in range(5):
            X, cls_preds[i], bbox_preds[i] = blk_forward(
                X,
                getattr(self, f'blk_{i}'),
                getattr(self, f'cls_{i}'),
                getattr(self, f'bbox_{i}')
            )
        cls_preds = concat_preds(cls_preds) # (bs, anchors*4)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes + 1) # (bs, anchors, (1+c))
        bbox_preds = concat_preds(bbox_preds) # (bs, anchors*4)
        return cls_preds, bbox_preds


# # ----------------------------------------------------------------
# #           搭建主干网络 shufflev1(我不会用, 效果不好, 难训练)
# # ----------------------------------------------------------------
# from torch.nn import functional as F
# class ShuffleV1Block(nn.Module):
#     def __init__(self, inp, oup, *, group, first_group, mid_channels, ksize, stride):
#         super(ShuffleV1Block, self).__init__()
#         self.stride = stride
#         assert stride in [1, 2]
#
#         self.mid_channels = mid_channels
#         self.ksize = ksize
#         pad = ksize // 2
#         self.pad = pad
#         self.inp = inp
#         self.group = group
#
#         if stride == 2:
#             outputs = oup - inp # 因为要与支路avgpool后的张量进行concat
#         else:
#             outputs = oup
#
#         branch_main_1 = [
#             # pw, point wise
#             nn.Conv2d(inp, mid_channels, 1, 1, 0, groups=1 if first_group else group, bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU6(inplace=True),
#             # dw, depth wise
#             nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
#             nn.BatchNorm2d(mid_channels),
#         ]
#         branch_main_2 = [
#             # pw-linear
#             nn.Conv2d(mid_channels, outputs, 1, 1, 0, groups=group, bias=False),
#             nn.BatchNorm2d(outputs),
#         ]
#         self.branch_main_1 = nn.Sequential(*branch_main_1)
#         self.branch_main_2 = nn.Sequential(*branch_main_2)
#
#         if stride == 2:
#             self.branch_proj = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
#
#     def forward(self, old_x):
#         x = old_x
#         x_proj = old_x
#         x = self.branch_main_1(x)
#         if self.group > 1:
#             x = self.channel_shuffle(x) # 通道混洗为什么加在这个位置? 噢加哪都没问题，因为是dw卷积，前后都行，那没事了
#         x = self.branch_main_2(x)
#         if self.stride == 1:
#             return F.relu6(x + x_proj)
#         elif self.stride == 2:
#             return torch.cat((self.branch_proj(x_proj), F.relu6(x)), 1)
#
#     def channel_shuffle(self, x):
#         batchsize, num_channels, height, width = x.data.size()
#         assert num_channels % self.group == 0
#         group_channels = num_channels // self.group
#
#         # 绝了，简单快速
#         x = x.reshape(batchsize, group_channels, self.group, height, width)
#         x = x.permute(0, 2, 1, 3, 4)
#         x = x.reshape(batchsize, num_channels, height, width)
#         return x
#
#
# def base_net(group):
#     """ 特征提取网络backbone.
#     :return: nn.Sequential
#     """
#     blk = []
#     num_c = [3, 24, 48, 96]
#     num_repeat = 3
#     blk.append(nn.Sequential(
#             nn.Conv2d(num_c[0], num_c[1], 3, 2, 1, bias=False),
#             nn.BatchNorm2d(num_c[1]),
#             nn.ReLU6(inplace=True),
#     ))
#
#     # 降采样: 混洗连块
#     input_channel = num_c[-3]
#     output_channel = num_c[-2]
#     for i in range(num_repeat):  # shuffleblock重复num_repeat次
#         stride = 2 if i == 0 else 1
#         first_group = i == 0
#         blk.append(ShuffleV1Block(input_channel, output_channel,
#                                   group=group, first_group=first_group,
#                                   mid_channels=num_c[-2] // 4, ksize=3, stride=stride))
#         input_channel = output_channel
#
#     # 降采样: 混洗连块
#     input_channel = num_c[-2]
#     output_channel = num_c[-1]
#     for i in range(num_repeat):  # shuffleblock重复num_repeat次
#         stride = 2 if i == 0 else 1
#         blk.append(ShuffleV1Block(input_channel, output_channel,
#                                   group=group, first_group=False,
#                                   mid_channels=num_c[-1] // 4, ksize=3, stride=stride))
#         input_channel = output_channel
#     return nn.Sequential(*blk)
#
#
# def shuffle_blk(in_channels, out_channels, group, num_repeat):
#     """ shuffle块
#     :param in_channels: 输入通道数
#     :param out_channels: 输出通道数
#     :param group:
#     :param num_repeat: 块重复次数
#     :return: nn.Sequential
#     """
#     blk = []
#     for i in range(num_repeat):
#         stride = 2 if i == 0 else 1
#         blk.append(ShuffleV1Block(in_channels, out_channels,
#                                   group=group, first_group=False,
#                                   mid_channels=out_channels // 4, ksize=3, stride=stride))
#         in_channels = out_channels
#     return nn.Sequential(*blk)
#
#
# def get_blk(i, in_channels, group, num_repeat):
#     """ 为了构造网络代码方便
#     :param i: index 0 1 2 3 4
#     :param in_channels:
#     :param group:
#     :param num_repeat:
#     :return:
#     """
#     if i == 0:
#         blk = base_net(group)
#     elif i == 4:
#         blk = nn.AdaptiveMaxPool2d((1, 1))
#     else:
#         blk = shuffle_blk(in_channels[i], in_channels[i+1], group, num_repeat)
#     return blk
#
#
# def blk_forward(X, blk, cls_predictor, bbox_predictor):
#     """ stage之间产生分支输出
#     :param X: 输入tensor
#     :param blk: 各个网络块blk
#     :param cls_predictor: Conv, 用于产生该blk输出的类别输出
#     :param bbox_predictor: Conv, 用于产生该blk输出的边框偏移输出
#     :return: 该层的(主干输出、类别分支、边界框偏移分支)
#     """
#     Y = blk(X)
#     cls_preds = cls_predictor(Y)
#     bbox_preds = bbox_predictor(Y)
#     return Y, cls_preds, bbox_preds
#
#
# class TinySSD(nn.Module):
#     def __init__(self, app, cn, group=3):
#         """ 总体网络
#         :param app: 每个像素点分配的锚框数(anchors per pixel)
#         :param cn: 类别总数(不包含背景)(class number)
#         """
#         super(TinySSD, self).__init__()
#         self.app = app
#         self.cn = cn
#
#         stage_repeats = [-1, 2, 2, 2, -1]
#         in_channels = [-1, 96, 120, 144, 168] # 改第二个数字，前面base_net也要改，其他随意改
#
#         for i in range(5):
#             setattr(self, f'blk_{i}', get_blk(i, in_channels, group, stage_repeats[i]))
#             if i != 4: # 主干网络、混洗块1、混洗块2、混洗块3的结果
#                 setattr(self, f'cls_{i}', gen_cls_predictor(in_channels[i+1], app, cn))
#                 setattr(self, f'bbox_{i}', gen_bbox_predictor(in_channels[i+1], app))
#             else: # 处理Global Average Pooling的结果
#                 setattr(self, f'cls_{i}', gen_cls_predictor(in_channels[-1], app, cn))
#                 setattr(self, f'bbox_{i}', gen_bbox_predictor(in_channels[-1], app))
#
#         self._initialize_weights()
#
#     def forward(self, x):
#         """ 神经网络前向传播
#         :param x: (bs, 3, w, h)
#         :return: (bs, anchors, 1+c), (bs, anchors*4)
#         """
#         cls_preds, bbox_preds = [None] * 5, [None] * 5
#         for i in range(5):
#             x, cls_preds[i], bbox_preds[i] = blk_forward(
#                 x,
#                 getattr(self, f'blk_{i}'),
#                 getattr(self, f'cls_{i}'),
#                 getattr(self, f'bbox_{i}')
#             )
#         cls_preds = concat_preds(cls_preds) # (bs, anchors*(1+c))
#         cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.cn + 1) # (bs, anchors, (1+c))
#         bbox_preds = concat_preds(bbox_preds) # (bs, anchors*4)
#         return cls_preds, bbox_preds
#
#     def _initialize_weights(self):
#         for name, m in self.named_modules():
#             if isinstance(m, nn.Conv2d):
#                 if 'first' in name:
#                     nn.init.normal_(m.weight, 0, 0.01)
#                 else:
#                     nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0001)
#                 nn.init.constant_(m.running_mean, 0)
#             elif isinstance(m, nn.BatchNorm1d):
#                 nn.init.constant_(m.weight, 1)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0.0001)
#                 nn.init.constant_(m.running_mean, 0)


if __name__ == '__main__':
    net = TinySSD(4, 1)
    from torchsummary import summary
    summary(net.cuda(), (3, 320, 320))
    # summary(net, (3, 320, 320), device='cpu') # 没有GPU用这条
