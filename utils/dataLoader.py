""" ヾ(•ω•`)o coding:utf-8
Auther: lee
Date: 2021/12/27
Time: 19:44:47
"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from utils.tools import cvtColor
import cv2


class MyDataSet(Dataset):
    def __init__(self, content_lines, img_r, mode):
        """
        :param content_lines: 处理后的数据的每一行：第一列是图像名字(绝对路径+后缀的那种)，后面跟4N个数字，表示目标坐上右下坐标，
        同一目标数组用逗号隔开，不同目标用空格隔开
        :param img_r: 网络的输入图片尺寸: int: 256
        :param mode: 'train' or 'test'
        """
        super(MyDataSet, self).__init__()
        self.content_lines = content_lines
        self.r = img_r
        self.len = len(content_lines)
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        index = index % self.len
        line = self.content_lines[index].split()

        # ------------------------------------------#
        #                  图像处理
        # ------------------------------------------#
        img = Image.open(line[0])
        img = cvtColor(img)
        iw, ih = img.size
        scale = min(self.r / iw, self.r / ih)
        nw = round(iw * scale)
        nh = round(ih * scale)
        dx = (self.r - nw) // 2  # 这俩有一个为0
        dy = (self.r - nh) // 2  # 这俩有一个为0

        img = img.resize((nw, nh), Image.BILINEAR)  # 双线性差值
        image_data = Image.new('RGB', (self.r, self.r), (128, 128, 128))
        image_data.paste(img, (dx, dy))

        flip = False
        if self.mode.lower() == 'train':
            # ------------------------------------------#
            #                  水平翻转
            # ------------------------------------------#
            flip = np.random.rand() > 0.5
            if flip: image_data = image_data.transpose(Image.FLIP_LEFT_RIGHT)

            # ------------------------------------------#
            #                  色域扭曲
            # ------------------------------------------#
            hue = 0.1
            sat = 1.3
            val = 1.3
            hue = self.rand(-hue, hue)
            sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
            val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
            x = cv2.cvtColor(np.array(image_data, np.float32) / 255, cv2.COLOR_RGB2HSV)
            # HSV三个通道依次为色调H、饱和度S、明度V H(0~360) S(0~1) V(0~1)
            x[..., 0] += hue * 360
            x[x[..., 0] > 360, 0] = 360
            x[x[..., 0] < 0, 0] = 0
            x[..., 1] *= sat
            x[..., 2] *= val
            x[..., 1:][x[..., 1:] > 1] = 1
            x[x < 0] = 0
            image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)

            # ------------------------------------------#
            #                  通道混洗
            # ------------------------------------------#
            channels = [0, 1, 2]
            np.random.shuffle(channels)
            image_data[:] = image_data[..., channels]


        _transform = transforms.Compose([
            transforms.ToTensor(),  # PIL int -> tensor[0~1] 浮点不归一化
        ])
        image_data = _transform(image_data)

        # ------------------------------------------#
        #                  边框处理
        # ------------------------------------------#
        # (o, 5) xmin xmax ymin ymax class
        boxes = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
        if len(boxes) > 0:
            np.random.shuffle(boxes)
            boxes = boxes.astype(np.float32)
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + dx
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + dy
            if flip: boxes[:, [0, 2]] = self.r - boxes[:, [2, 0]]
            boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
            boxes[:, 2][boxes[:, 2] > self.r] = self.r
            boxes[:, 3][boxes[:, 3] > self.r] = self.r
            box_w = boxes[:, 2] - boxes[:, 0]  # 左上右下 -> 中心宽高
            box_h = boxes[:, 3] - boxes[:, 1]  # 左上右下 -> 中心宽高
            boxes = boxes[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
            boxes[:, :4] = boxes[:, :4] / self.r  # 边界框大小被归一化为0~1

        # image_data: (3, 256, 256), boxes: (o, 5) or (0,)
        # boxes: cx cy w h c
        # image_data -> tensor boxes -> array
        return image_data, boxes

    @staticmethod
    def rand(a=0., b=1.):
        return np.random.rand() * (b - a) + a


def dataset_collate(batch):
    """ DataLoader中collate_fn使用
    之所以需要传入这个参数，是因为dataloader返回img和boxes，但是每个图片的boxes不固定，所以要外面封装一层，使之形状固定
    :param batch: list(tuple(tensor(3, 255, 255), ndarray(o, 5) or 没目标: ndarray([])))
    :return: (bs, 3, 255, 255) (bs, 100, 5) 5: class xmin ymin xmax ymax
    """
    m = 100 # 限制一张图片最多存在100个目标
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        if box.ndim == 1: # 如果没有目标
            bboxes.append(torch.full((m, 5), -1, dtype=torch.float32))
        else: # 当至少有一个目标时
            box[:] = box[:, [4,0,1,2,3]] # 把类别放在第一位
            bboxes.append(torch.from_numpy(np.pad(box, ((0,m-box.shape[0]),(0,0)), constant_values=-1)))
    return torch.stack(images, 0), torch.stack(bboxes, 0)


if __name__ == '__main__':
    from utils.tools import get_classes, show_bboxes
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    input_shape = 256
    class_names, _ = get_classes('../model_data/voc_classes.txt')
    with open('../2077_train.txt') as f:
        train_lines = f.readlines()

    train_dataset = MyDataSet(train_lines, input_shape, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,
                              pin_memory=True, drop_last=True, collate_fn=dataset_collate)
    dataiter = iter(train_loader)

    fig = plt.figure(figsize=(16, 8))
    columns = 4
    rows = 2
    inputs, labels = dataiter.next()
    for idx in range(columns * rows):
        ax = fig.add_subplot(rows, columns, idx + 1, xticks=[], yticks=[])
        label_boxes = labels[idx, labels[idx][:, 0] >= 0]
        show_bboxes(ax, label_boxes[:,1:]*input_shape, [class_names[int(item)] for item in label_boxes[:,0].tolist()])
        plt.imshow(inputs[idx].permute(1,2,0))

















