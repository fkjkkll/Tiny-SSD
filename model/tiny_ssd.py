""" ヾ(•ω•`)o coding:utf-8
Auther: lee
Date: 2022/01/02
Time: 13:49:16
"""
import torch
from torch.nn import functional as F
from model.anchor_generate import generate_anchors
from model.for_inference import multibox_detection
from model.net import TinySSD
from utils.tools import get_classes, get_anchor_info, try_gpu, img_preprocessing
import os

class Tiny_SSD(object):
    _defaults = {
        "anchor_sizes_path": 'model_data/anchor_sizes.txt',
        "anchor_ratios_path": 'model_data/anchor_ratios.txt',
        "model_path": 'model_data/result.pt',
        "classes_path": 'model_data/voc_classes.txt',
        "r": 256,
        "nms_threshold": 0.1,
    }

    @classmethod
    def get_default(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, isTraining=False, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        if isTraining:
            pass
        else:
            self.sizes = get_anchor_info(self.anchor_sizes_path)
            self.ratios = get_anchor_info(self.anchor_ratios_path)

            # -----------------------------------------------
            #                   产生先验锚框
            # -----------------------------------------------
            if len(self.sizes) != len(self.ratios):
                self.ratios = [self.ratios[0]] * len(self.sizes)
            self.anchors_perpixel = len(self.sizes[0]) + len(self.ratios[0]) - 1
            feature_map = [32, 16, 8, 4, 1]
            self.anchors = generate_anchors(feature_map, self.sizes, self.ratios)

            # -----------------------------------------------
            #                   加载网络
            # -----------------------------------------------
            self.name_classes, self.num_classes = get_classes(self.classes_path)
            self.device, self.net = try_gpu(), TinySSD(anchors_perpixel=self.anchors_perpixel, num_classes=self.num_classes)
            self.net.load_state_dict(torch.load(self.model_path))

            # 放入GPU，开启评估模式
            self.anchors = self.anchors.to(self.device)
            self.net = self.net.to(self.device)
            self.net.eval()


    def inference(self, image):
        """ 传入一副Image，输出预测结果
        :param image: PIL Image
        :return: list -> (chose, 10): class conf bbx bby bbx bby anx any anx any
        """
        iw, ih = image.size  # 返回的居然是(w, h)
        image = img_preprocessing(image).unsqueeze(0)  # (1, 3, 255, 255)
        with torch.no_grad():
            cls_preds, bbox_preds = self.net(image.to(self.device))  # (1, 5444, 2) (1, 5444*4)
            cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)  # (1, 2, 5444)
            output = multibox_detection(cls_probs, bbox_preds, self.anchors, self.nms_threshold)  # (1, 5444, 10)
            idx = [i for i, row in enumerate(output[0]) if row[0]>=0] # 排除背景
            result = output[0, idx]  # (chose, 10): class conf bbx bby bbx bby anx any anx any

            # 下面是将预测框还原回图像原有大小
            scale = min(self.r / ih, self.r / iw)
            dx = (self.r - round(iw * scale)) // 2  # 这俩有一个为0
            dy = (self.r - round(ih * scale)) // 2  # 这俩有一个为0
            result[:, 2:] *= 256
            result[:, [2, 4, 6, 8]] -= dx # bbxmin bbxmax anxmin anxmax
            result[:, [3, 5, 7, 9]] -= dy # bbymin bbymax anymin anymax
            result[:, 2:] /= scale
            result[result<0] = 0
            result[result[:, 4] > iw, 4] = iw # bbxmax
            result[result[:, 5] > ih, 5] = ih # bbymax
        return result


    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        output = self.inference(image) # (chose, 10)
        for i in range(len(output)):
            predicted_class = self.name_classes[int(output[i,0])]
            if predicted_class not in class_names: continue # 排除不要的类
            score = str(float(output[i, 1]))
            xmin,ymin,xmax,ymax = output[i, 2:6]
            f.write("%s %s %s %s %s %s\n" % (
                predicted_class, score[:6], str(int(xmin)), str(int(ymin)), str(int(xmax)), str(int(ymax))
            ))
        f.close()
        return

















