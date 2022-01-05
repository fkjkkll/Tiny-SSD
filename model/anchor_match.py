""" ヾ(•ω•`)o coding:utf-8
Auther: lee
Date: 2021/12/28
Time: 14:26:00
"""
import torch

def box_corner_to_center(boxes):
    """ 将锚框从(minx, miny, maxx, maxy)转换为(centerx, centery, width, height)
    :param boxes: (n, 4)
    :return: (n, 4)
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes


def box_center_to_corner(boxes):
    """ 将锚框从(centerx, centery, width, height)转换为(minx, miny, maxx, maxy)
    :param boxes: (n, 4)
    :return: (n, 4)
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes


def box_iou(boxes1, boxes2):
    """ 计算两组boxs之间的IOU
    :param boxes1: (n, 4) n指预设的anchor的数量(anchors)
    :param boxes2: (m, 4) m指预设的一幅图片最大目标数(o)
    :return: (n, m) 每个位置代表i,j对应的两个框的交并比
    """
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))  # 计算面积
    areas1 = box_area(boxes1) # (anchors,)
    areas2 = box_area(boxes2) # (o,)

    # 交集(利用了广播机制)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)  # 不相交会有负值，设为0
    inter_areas = inters[:, :, 0] * inters[:, :, 1] # 相交部分的面积，[:, :, 0]相对于相交的宽，[:, :, 1]相对于相交的高

    # 并集(利用了广播机制)
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas


def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """ 为一些锚框分配GT，iou不够就不分配
    :param ground_truth: (o, 4) o是该图片中的目标数
    :param anchors: (anchors, 4)
    :param device:
    :param iou_threshold: 表示给锚框分配正样本的阈值
    :return: (anchors,) 表示对应锚框分配的gt框的下标，没有分配的是-1
    """
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]

    # 位于第i行和第j列的元素 x_ij 是锚框i和真实边界框j的IoU
    jaccard = box_iou(anchors, ground_truth) # (anchors, o)

    # 记录给每个锚框分配的真实目标(0~o-1)的下标， 没分配就是-1
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device) # (anchors,)

    # ------------------------ 先为大于IOU阈值的anchor分配gt --------------------------
    # 根据阈值，决定是否分配真实边界框 alloc_gt值(0 ~ o-1)o是该图片真实存在多少目标，大部分都是0(因为没有与目标交集，IOU=0，取max就是第一位:0)
    max_ious, alloc_gt = torch.max(jaccard, dim=1) # (anchors,) (anchors,)
    active_index = max_ious >= iou_threshold
    anchors_bbox_map[active_index] = alloc_gt[active_index]

    # ------------------------ 再为每个gt分配最适合它的 --------------------------
    col_discard = torch.full((num_anchors,), -1) # 用于替换的
    row_discard = torch.full((num_gt_boxes,), -1) # 用于替换的
    for idx in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard) # 不指定dim则用一维数组下标形式返回多维数组结果
        gt_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = gt_idx
        jaccard[:, gt_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map


def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """ 给定锚框和分配给它的GT，计算偏移值（希望网络输出的偏移值）
    :param anchors: (anchors, 4)
    :param assigned_bb: (anchors, 4)
    :param eps:
    :return: (anchors, 4) offset_x offset_y offset_w offset_h
    """
    c_anchors = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anchors[:, :2]) / c_anchors[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anchors[:, 2:])
    offset = torch.cat((offset_xy, offset_wh), axis=1)
    return offset


def multibox_target(anchors, labels):
    """ 使用真实边界框标记锚框(为某些锚框匹配GT，返回这些锚框应该的（锚框偏移量、掩膜、类别），属于转换后的最终label)
    :param anchors: (anchors, 4)
    :param labels: (bs, 100, 5) 100是该副图片中预设的最多有多少目标 5是 class minx miny maxx maxy
    :return: 一个元组，包含3个部分：
    bbox_offset: (bs, anchors*4)
    bbox_mask: (bs, anchors*4) 里面元素非0即1
    class_labels: (bs, anchors) 代表着类别，0是背景
    """
    batch_size = labels.shape[0]
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size): # 拿出batch中一个一个img处理
        label = labels[i, :, :]
        label = label[label[:, 0] >= 0] # 取出该图片中含有目标的label(某些行，可能为0:表示不含有任何目标) (!!!)
        if len(label) == 0: # 如果不含有任何目标
            batch_offset.append(torch.zeros((num_anchors*4,), dtype=torch.float32, device=device))
            batch_mask.append(torch.zeros((num_anchors*4,), dtype=torch.float32, device=device))
            # indices must be long, byte or bool tensors
            batch_class_labels.append(torch.zeros((num_anchors,), dtype=torch.long, device=device))
        else: # 当含有目标时
            anchor_map_object = assign_anchor_to_bbox(label[:, 1:], anchors, device) # (anchors,)
            bbox_mask = ((anchor_map_object >= 0).float().unsqueeze(-1)).repeat(1, 4) # (anchors, 4)

            # 初始化[分配的类别]和[分配的边界框坐标]: 所以0是背景类
            assigned_cls = torch.zeros(num_anchors, dtype=torch.long, device=device) # (anchors,)
            assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device) # (anchors, 4)

            # 使用真实边界框来标记锚框的类别
            # 如果一个锚框没有被分配，我们标记其为背景（值为0）
            active_indices = torch.nonzero(anchor_map_object >= 0) # (active_anchors, 1)
            assigned_object_idx = anchor_map_object[active_indices] # (active_anchors, 1)
            assigned_cls[active_indices] = label[assigned_object_idx, 0].long() + 1 # (anchors,) 背景类是0，txt中0类变为1 (!!!)
            assigned_bb[active_indices] = label[assigned_object_idx, 1:] # (anchors, 4)

            # 偏移量转换 (预测是: cx cy w h的偏移量)
            # 这里面只有active_anchors个锚框是有效的，用的时候要乘以bbox_mask (!!!)
            offset = offset_boxes(anchors, assigned_bb)

            batch_offset.append(offset.reshape(-1)) # (anchors*4,)
            batch_mask.append(bbox_mask.reshape(-1)) # (anchors*4,)
            batch_class_labels.append(assigned_cls) # (anchors,)

    # (bs, anchors*4) (bs, anchors*4) (bs, anchors)
    return torch.stack(batch_offset), torch.stack(batch_mask), torch.stack(batch_class_labels)