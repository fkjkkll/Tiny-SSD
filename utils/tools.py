""" ヾ(•ω•`)o coding:utf-8
Auther: lee
Date: 2021/12/27
Time: 20:54:03
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

def get_classes(path):
    """ 获得类别和类别数目
    :param path: 文件路径
    :return: list_classes, num_classes
    """
    with open(path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def get_anchor_info(path):
    """ 获得锚框的信息(网络有五个尺度预测)
    :param path:  文件路径
    :return: 二维列表
    """
    with open(path, encoding='utf-8') as f:
        anchor_infos = f.readlines()
    anchor_infos = [list(map(float, c.strip().split())) for c in anchor_infos]
    return anchor_infos


def cvtColor(image):
    """ 将图像转换成RGB图像
    :param image: image
    :return: RGBimage
    """
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def img_preprocessing(_img):
    """ 用于inference，Image(H,W,C) -> (C,H,W) 0~1
    :param _img: Image
    :return: tensor_img (C,H,W) 0~1
    """
    r = 256
    if hasattr(_img, 'size'): iw, ih = _img.size #
    else: iw, ih, _ = _img.shape
    scale = min(r / iw, r / ih)
    nw = round(iw * scale)
    nh = round(ih * scale)
    dx = (r - nw) // 2 # 这俩有一个为0
    dy = (r - nh) // 2 # 这俩有一个为0
    _img = _img.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (r, r), (128, 128, 128))
    new_image.paste(_img, (dx, dy))
    transform = transforms.Compose([
        transforms.ToTensor(), # PIL -> tensor [0~1]
    ])
    return transform(new_image)


def show_bboxes(axes, bboxes, labels=None, colors=None):
    """ 在子图中绘制锚框 xmin xmax ymin ymax
    :param axes: 传来的画布
    :param bboxes: (num_anchors, 4)
    :param labels: default = None
    :param colors: default = None
    :return: None
    """
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0],
                             height=bbox[3] - bbox[1], fill=False, edgecolor=color, linewidth=1)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i], va='center', ha='center',
                      fontsize=9, color=text_color, bbox=dict(facecolor=color, lw=0))


def try_gpu():
    """ 尝试用GPU，没有则返回CPU
    :return: device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class Timer:
    """ Record multiple running times"""
    def __init__(self):
        self.tik = None
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    # python语法: 魔法方式一般以双下划线开头和结尾
    def __getitem__(self, idx):
        return self.data[idx]


def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

class Animator:
    """For plotting data in animation.(Incrementally plot multiple lines)"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=0, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, figsize=(8, 6)):
        if legend is None: legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts
        plt.pause(0.1) # 由于单线程，不这样会卡住


    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla() # 清除子图
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes() # 之所以写成lambda，估计是不想在这里传参数
        plt.pause(0.1) # 由于单线程，不这样会卡住




