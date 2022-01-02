import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from model.tiny_ssd import Tiny_SSD
from utils.tools import get_classes
from utils.utils_map import get_map

if __name__ == "__main__":
    '''
    Recall和Precision不像AP是一个面积的概念，在门限值不同时，网络的Recall和Precision值是不同的。
    map计算结果中的Recall和Precision代表的是当预测时，门限置信度为0.5时，所对应的Recall和Precision值。

    此处获得的./map_out/detection-results/里面的txt的框的数量会比直接predict多一些，这是因为这里的门限低，
    目的是为了计算不同门限条件下的Recall和Precision值，从而实现map的计算。
    '''
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode用于指定该文件运行时计算的内容
    #   map_mode为0代表仅仅获得真实框。
    #   map_mode为1代表[获得预测结果、计算VOC_map]
    #   map_mode为2代表仅仅获得预测结果。
    #   map_mode为3代表仅仅计算VOC_map。
    #-------------------------------------------------------------------------------------------------------------------#
    mode        = 1
    #-------------------------------------------------------#
    #   此处的classes_path用于指定需要测量VOC_map的类别
    #   一般情况下与训练和预测所用的classes_path一致即可
    #-------------------------------------------------------#
    classes_path    = 'model_data/voc_classes.txt'
    #-------------------------------------------------------#
    #   MINOVERLAP用于指定想要获得的mAP0.x
    #   比如计算mAP0.75，可以设定MINOVERLAP = 0.75
    #-------------------------------------------------------#
    MINOVERLAP      = 0.5
    #-------------------------------------------------------#
    #   map_vis用于指定是否开启VOC_map计算的可视化
    #-------------------------------------------------------#
    map_vis         = True
    conf_threshold  = 0.3
    #-------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    year = '2077'
    #-------------------------------------------------------#
    #   结果输出的文件夹，默认为result
    #-------------------------------------------------------#
    map_out_path    = 'result'

    image_ids = open(os.path.join(VOCdevkit_path, f"VOC{year}/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if mode == 0:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, f"VOC{year}/Annotations/"+image_id+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult') is not None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox  = obj.find('bndbox')
                    left    = bndbox.find('xmin').text
                    top     = bndbox.find('ymin').text
                    right   = bndbox.find('xmax').text
                    bottom  = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if mode == 1 or mode == 2:
        print("Load model ...")
        net = Tiny_SSD()

        print("Get predict result ...")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, f"VOC{year}/JPEGImages/"+image_id+".jpg")
            image       = Image.open(image_path)
            # if map_vis:
            #     image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
            net.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")

    if mode == 1 or mode == 3:
        print("Get map ...")
        get_map(MINOVERLAP, True, path = map_out_path)

        print("Save image results ...")
        if map_vis:
            from PIL import ImageDraw, ImageFont
            font = ImageFont.truetype('Sundries/simhei.ttf', 20)
            for image_id in tqdm(image_ids):
                image_path = os.path.join(VOCdevkit_path, f"VOC{year}/JPEGImages/" + image_id + ".jpg")
                image = Image.open(image_path)
                painter = ImageDraw.ImageDraw(image)

                with open(os.path.join(map_out_path, f'ground-truth/{image_id}.txt')) as f:
                    gt_lines = f.readlines()
                with open(os.path.join(map_out_path, f'detection-results/{image_id}.txt')) as f:
                    pred_lines = f.readlines()

                for i, line in enumerate(gt_lines): # 画ground-truth框
                    line = line.split()
                    x1, y1, x2, y2 = list(map(int, line[1:]))
                    painter.rectangle(((x1, y1), (x2, y2)), fill=None, outline='red', width=5)
                    painter.text((x1, y1), line[0], font=font)
                for i, line in enumerate(pred_lines): # 画预测框
                    line = line.split()
                    if float(line[1]) < conf_threshold: continue
                    x1, y1, x2, y2 = list(map(int, line[2:]))
                    painter.rectangle(((x1, y1), (x2, y2)), fill=None, outline='blue', width=1)
                    painter.text((x1, y1), f'{line[0]}:[{line[1]}]', font=font)

                image.save(os.path.join(map_out_path, f'images-optional/{image_id}.jpg'))
        print('Done.')



























