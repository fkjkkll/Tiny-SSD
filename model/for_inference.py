""" ヾ(•ω•`)o coding:utf-8
Auther: lee
Date: 2021/12/28
Time: 20:01:46
"""
from model.anchor_match import *
import torch

def offset_inverse(anchors, offset_preds):
    """ [预测的偏移量]结合[锚框]得出[预测的目标框]
    :param anchors: (num_anchors, 4)
    :param offset_preds: (num_anchors, 4)
    :return: (num_anchors, 4)
    """
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox


def nms(boxes, scores, iou_threshold):
    """ 非极大值抑制: 对预测边界框的置信度进行排序
    :param boxes: (num_anchors, 4)
    :param scores: (num_anchors,)
    :param iou_threshold:
    :return: 保留下来的boxes的下标
    """
    B = torch.argsort(scores, dim=-1, descending=True) # 返回排序后的下标
    keep = []  # 保留预测边界框的指标
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break

        # 计算 [当前循环最大置信度的锚框] 与 [当前集合中的所有锚框] 的iou, 结果转成1维向量
        iou = box_iou(boxes[i, :].reshape(-1, 4), boxes[B[1:], :].reshape(-1, 4)).reshape(-1) # 不区分类的NMS (!!!)

        # 大于iou阈值的排除，剩下的锚框的下标
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)

        # 之所以加1，是因为送入box_iou函数的两个boxes是0和[1:]，但是送入函数得出的结果都是从0开始
        # 加1才能在box_iou函数结束后还原[1:]的位置
        B = B[inds + 1]  # 没错，即便inds为空，也可，也即，下标可以用一个空的list来索引，结果也是tensor([])
    return torch.tensor(keep, device=boxes.device)


def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5, object_threshold=0.1, max_object_num=100):
    """ 将网络的输出值转化为我们需要的信息
    :param cls_probs: (bs, classes, anchors) -> (bs, 1+c, anchors) 背景+类别个数
    :param offset_preds: (bs, anchors*4) -> (bs, 4 * anchors)
    :param anchors: (anchors, 4)
    :param nms_threshold: NMS阈值
    :param object_threshold: 预设的认为是目标的阈值
    :param max_object_num: 预设的最大输出目标数量
    :return: (bs, max_object_num, 10) class conf minx miny maxx maxy aminx aminy amaxx amaxy 除了类别都是0~1
    """
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4) # ((1+c), anchors) (anchors, 4)

        # 取每个锚框对不同类的最大置信那个, 从1行开始表示不要背景(0行), conf大部分是很小的数(大数字在0行，指背景), 单类识别的话id全是0(因为就1行)
        conf, class_id = torch.max(cls_prob[1:], axis=0) # (anchors,) (anchors,)

        # chose_idx指代的下标是 0~anchors-1
        chose_idx = torch.nonzero(conf>object_threshold).reshape(-1) # (chose1,) 第一次筛选

        predicted_bb = offset_inverse(anchors, offset_pred) # (chose1, 4)
        keep = nms(predicted_bb[chose_idx], conf[chose_idx], nms_threshold) # (chose2,)
        chose_idx = chose_idx[keep] # nms后，chose_idx已经按照conf降序排序 (chose2,) 第二次筛选

        if len(chose_idx) > max_object_num: # 因为要凑够一个固定形状批量输出 [大于要截断]
            chose_idx = chose_idx[:max_object_num]

        class_id = class_id[chose_idx].unsqueeze(1) # (chose2, 1)
        conf = conf[chose_idx].unsqueeze(1) # (chose2, 1)
        predicted_bb = predicted_bb[chose_idx] # (chose2, 4)
        choosed_anchors = anchors[chose_idx] # # (chose2, 4)

        if len(chose_idx) < max_object_num: # 因为要凑够一个固定形状批量输出 [不够要填充]
            supplement_num = max_object_num - len(chose_idx)
            class_id = torch.cat((class_id, torch.full((supplement_num, 1), -1, device=device)), dim=0) # 多余填充-1(背景)
            conf = torch.cat((conf, torch.zeros((supplement_num, 1), device=device)), dim=0)
            predicted_bb = torch.cat((predicted_bb, torch.zeros((supplement_num, 4), device=device)), dim=0)
            choosed_anchors = torch.cat((choosed_anchors, torch.zeros((supplement_num, 4), device=device)), dim=0)

        # 组装结果
        pred_info = torch.cat((class_id, conf, predicted_bb, choosed_anchors), dim=1)
        out.append(pred_info)
    return torch.stack(out) # (bs, max_object_num, 10)