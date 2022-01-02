""" ヾ(•ω•`)o coding:utf-8
Auther: lee
Date: 2021/12/27
Time: 18:06:06
"""
from utils.tools import *
from PIL import Image
from model.tiny_ssd import Tiny_SSD

# -----------------------------------------------
#                    画图函数
# -----------------------------------------------
def showres(_image, _output, _threshold, _name_classes):
    """ 显示图片和边界框
    :param _image: PIL Image
    :param _output: (chose, 10) 真实图片大小
    :param _threshold: 置信度阈值
    :param _name_classes: 类别列表
    :return:
    """
    plt.cla() # 清除上一刻图
    plt.draw() # 会刷新图
    fig = plt.imshow(_image)
    for row in _output:
        score = float(row[1])
        _class = _name_classes[row[0].long()]
        if score < _threshold: continue
        bbox = [row[2:6]]
        abox = [row[6:10]]
        show_bboxes(fig.axes, abox, f'{_class}[{score:.2f}]', 'w')  # 显示对应的anchors
        show_bboxes(fig.axes, bbox, f'{_class}[{score:.2f}]', 'b')  # '%.2f'%value格式化输出


# -----------------------------------------------
#                     摄像头
# -----------------------------------------------
import cv2
cap = cv2.VideoCapture(0)
ssd = Tiny_SSD()
name_classes, num_classes = ssd.name_classes, ssd.num_classes
device = ssd.device

try:
    while True:
        _, frame = cap.read() # h w c
        frame[:] = frame[..., [2, 1, 0]]  # cv2的图像都是BGR，需要转换为RGB
        frame = Image.fromarray(frame, mode='RGB')
        output = ssd.inference(frame)
        showres(frame, output, 0.3, ssd.name_classes)
        plt.pause(0.001)
except KeyboardInterrupt:
    cap.release()


# -----------------------------------------------
#                     图片
# -----------------------------------------------
# image = Image.open('VOCdevkit/test.jpg').convert('RGB')
# ssd = Tiny_SSD()
# output = ssd.inference(image)
# showres(image, output, 0.3, ssd.name_classes)









