""" ヾ(•ω•`)o coding:utf-8
Auther: lee
Date: 2021/12/28
Time: 13:56:55
"""
import torch

def multibox_prior(data, sizes, ratios):
    """ 为特征图按像素生成锚框: 生成以每个像素为中心具有不同形状的锚框
    注意: 在实践中，我们只考虑包含 s1 或 r1 的组合，即sizes[0]与ratios的所有组合加上ratios[0]与sizes的所有组合
    去除一个重复，一个像素点总共有s+r-1个锚框，例如s=2，r=3，即有4个锚框
    :param data: (batch_size, channels, fw, fh) 即网络流通中的特征图，一般预设锚框时batch_size = channels = 1
    :param sizes: 一个列表，放入不同边长的缩放比例
    :param ratios: 一个列表，放入不同的宽高比（相对于原图）
    :return: (fw*fh*(len(s)+len(r)-1), 4)，即对每个特征图的一个“像素”产生一系列锚框
    """
    in_height, in_width = data.shape[-2:]  # 从倒数第二个开始到最后一个，也即最后俩
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为了将锚点移动到像素的中心，需要设置偏移量。
    # 因为一个像素的的高为1且宽为1，我们选择偏移我们的中心0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in yaxis
    steps_w = 1.0 / in_width  # Scaled steps in xaxis

    # 生成锚框的所有中心点
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h # (fh,) 0~1
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w # (fw, ) 0~1
    shift_y, shift_x = torch.meshgrid(center_h, center_w,  indexing='ij')  # (fh, fw), (fh, fw) 0~1
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)  # 拉成一维，两者搭配zip即代表每个锚框的中心坐标(0~1)

    # 生成“boxes_per_pixel”个高和宽, (4,) 0~1
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]), sizes[0] * torch.sqrt(ratio_tensor[1:])))
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]), sizes[0] / torch.sqrt(ratio_tensor[1:])))

    # (fw*fh*bpp, 4)，其中除以2来获得半高和半宽->从宽高转换为坐标
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2
    # 里面的4个值的意义是，以原点为中心，锚框的xmin xmax ymin ymax，下一步是将锚框放到正确的位置

    # (fw*fh*bpp, 4)
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    # 里面的4个值的意义是，xcenter ycenter xcenter ycenter也就是说1、3列重复 2、4列重复仅为了方便
    '''
    repeat和repeat_interleave区别是，比如1,2,3，重复3次
    repeat: 1,2,3,1,2,3,1,2,3 [按块重复]
    repeat和repeat_interleave区别是: 1,1,1,2,2,2,3,3,3 [内部重复]
    '''
    output = out_grid + anchor_manipulations
    return output

def generate_anchors(fmp_list, sizes, ratios):
    """ 针对特征图生成一系列锚框
    :param fmp_list: 各层输出的特征图大小(只能是方图)，例如[32,16,8,4,1]
    :param sizes: 各层锚框尺度，例如[[0.1, 0.15], [0.25, 0.3], [0.35, 0.4], [0.5, 0.6], [0.7, 0.9]]
    :param ratios: 各层锚框长宽比，例如[[1, 2, 0.5]] * 5
    :return: 汇合所有锚框组成(N, 4)，例如tensor.shape = (anchors, 4):
    """
    anchors = [None]*len(fmp_list)
    for i, fmp in enumerate(fmp_list):
        tmp = torch.zeros((1,1,fmp,fmp))
        anchors[i] = multibox_prior(tmp, sizes[i], ratios[i])
    anchors = torch.cat(anchors, dim=0)
    return anchors

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from utils.tools import *

    r = get_image_size('../model_data/image_size.txt')
    img = Image.open('../VOCdevkit/test.jpg').convert('RGB')
    img = img_preprocessing(img)
    fh, fw = 1, 1

    X = torch.rand(size=(1, 1, fh, fw))
    anchor_sizes = get_anchor_info('../model_data/anchor_sizes.txt')
    anchor_ratios = get_anchor_info('../model_data/anchor_ratios.txt')
    anchors_boxes = multibox_prior(X, anchor_sizes[4], anchor_ratios[0])

    index = np.random.choice(fw*fh*4, 50)
    fig = plt.imshow(img.permute(1,2,0))
    show_bboxes(fig.axes, anchors_boxes[index] * r)

    pass
