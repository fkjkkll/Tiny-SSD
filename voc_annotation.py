""" ヾ(•ω•`)o coding:utf-8
Auther: lee
Date: 2021/12/27
Time: 19:58:30
"""
import os
import random
import xml.etree.ElementTree as ET

from utils.tools import get_classes

# -------------------------------------------------------------------#
#   trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1
#   train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1
# -------------------------------------------------------------------#
trainval_percent = 0.9
train_percent = 0.9

classes_path = 'model_data/voc_classes.txt'
classes, _ = get_classes(classes_path)

VOCdevkit_path = 'VOCdevkit'
VOCdevkit_sets = [('2077', 'trainval'), ('2077', 'test')]

def convert_annotation(_year, _image_id, _list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml' % (_year, _image_id)), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') is not None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
             int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        _list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets\Main ...")
    xmlfilepath = os.path.join(VOCdevkit_path, f'VOC{VOCdevkit_sets[0][0]}/Annotations')
    saveBasePath = os.path.join(VOCdevkit_path, f'VOC{VOCdevkit_sets[0][0]}/ImageSets/Main')
    temp_xml = os.listdir(xmlfilepath)
    total_xml = []
    for xml in temp_xml:
        if xml.endswith(".xml"):
            total_xml.append(xml)

    num = len(total_xml)
    itemList = range(num)

    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(itemList, tv)
    train = random.sample(trainval, tr)

    print("train and val size", tv)
    print("train size", tr)
    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

    for i in itemList:
        name = total_xml[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print("Generate txt in ImageSets done.")

    print(f"Generate .txt for train ...")
    for year, image_set in VOCdevkit_sets:
        # image_ids saved the name of imgs(or annotations)(don't include suffix)
        image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt' % (year, image_set)),
                         encoding='utf-8').read().strip().split()
        list_file = open('%s_%s.txt' % (year, image_set), 'w', encoding='utf-8')
        for image_id in image_ids:
            list_file.write('%s/VOC%s/JPEGImages/%s.jpg' % (os.path.abspath(VOCdevkit_path), year, image_id))
            convert_annotation(year, image_id, list_file)
            list_file.write('\n')
        list_file.close()
    print("done\n")
