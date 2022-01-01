""" ヾ(•ω•`)o coding:utf-8
Auther: lee
Date: 2021/12/27
Time: 18:06:06
"""
import matplotlib.pyplot as plt

from utils.tools import *
from model.anchor_generate import generate_anchors
from model.net import TinySSD
from model.loss import *
from PIL import Image
from model.for_inference import multibox_detection

# -----------------------------------------------
#                   产生先验锚框
# -----------------------------------------------
sizes = get_anchor_info('model_data/anchor_sizes.txt')
ratios = get_anchor_info('model_data/anchor_ratios.txt')
if len(sizes) != len(ratios): ratios = [ratios[0]] * len(sizes)
anchors_perpixel = len(sizes[0]) + len(ratios[0]) - 1
feature_map = [32,16,8,4,1]
anchors = generate_anchors(feature_map, sizes, ratios)

# -----------------------------------------------
#                   加载网络
# -----------------------------------------------
classes_path = 'model_data/voc_classes.txt'
name_classes, num_classes = get_classes(classes_path)
device, net = try_gpu(), TinySSD(anchors_perpixel=anchors_perpixel, num_classes=num_classes)
net.load_state_dict(torch.load('model_data/result.pt'))

# -----------------------------------------------
#                      推理
# -----------------------------------------------
def predict(_net, X):
    """ 预测
    :param _net: trained network
    :param X: (bs, 3, imgh, imgw)
    :return:
    """
    _net.eval()
    cls_preds, bbox_preds = _net(X.to(device)) # (1, 5444, 2) (1, 5444*4)
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1) # (1, 2, 5444)
    output = multibox_detection(cls_probs, bbox_preds, anchors, 0.3) # (1, 5444, 10)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1] # list 找到非背景类
    return output[0, idx] # (chose, 10)


def showres(X, output, threshold):
    """ 显示图片和边界框
    :param X:
    :param output:
    :param threshold:
    :return:
    """
    plt.cla() # 清除上一刻图
    plt.draw() # 会刷新图
    fig = plt.imshow(X)
    for row in output:
        score = float(row[1])
        _class = name_classes[row[0].long()]
        if score < threshold:
            continue
        h, w = X.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        abox = [row[6:10] * torch.tensor((w, h, w, h), device=row.device)] # 显示对应的anchors
        show_bboxes(fig.axes, abox, f'{_class}[{score:.2f}]', 'w')  # 显示对应的anchors
        show_bboxes(fig.axes, bbox, f'{_class}[{score:.2f}]', 'b')  # '%.2f'%value格式化输出


# ------------------------------------- 摄像头 -------------------------------------
import cv2
anchors = anchors.to(device)
net = net.to(device)
cap = cv2.VideoCapture(0)

try:
    while True:
        _, frame = cap.read()
        frame = Image.fromarray(frame, mode='RGB')
        frame = img_preprocessing(frame)
        frame[:] = frame[[2, 1, 0]]  # cv2 BGR -> RGB
        frame = frame.to(device)
        res = predict(net, frame.unsqueeze(0))
        showres(frame.cpu().permute(1, 2, 0), res.cpu(), threshold=0.3)
        plt.pause(0.001)
except KeyboardInterrupt:
    cap.release()


# ------------------------------------- 图片 -------------------------------------
# img = Image.open('VOCdevkit/test.jpg').convert('RGB')
# img = img_preprocessing(img)
# anchors = anchors.to(device)
# net = net.to(device)
# img = img.to(device)
# res = predict(net, img.unsqueeze(0))
# showres(img.cpu().permute(1,2,0), res.cpu(), threshold=0.3)








