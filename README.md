## Tiny-SSD
An implementation of SSD

## Usage
1. Download voc2007 or voc2012 or voc2077(My small toy face detection dataset, you can use any dataset which follows the VOC structure)
2. Put them in the right directory
3. Run voc_annotation to generate divided dataset(in VOCdevkit\VOC20..\ImageSets\Main) and dataset labels(in code's main directory)
4. Run train.py to train a simple model(will not consume too many time). It will save .pt weights in model_date/ every epoch automatically
5. inference or evaluation(config info can be modified in model/tiny_ssd.py):
    1. run inference.py to get object detection result from a picture or your pc camera
    2. run evaluation.py to evaluate your test set


## Result(My small toy face detection results)
![结果1](https://github.com/fkjkkll/Tiny-SSD/blob/master/Sundries/result1.jpg)
![结果2](https://github.com/fkjkkll/Tiny-SSD/blob/master/Sundries/result2.jpg)
![结果3](https://github.com/fkjkkll/Tiny-SSD/blob/master/Sundries/result3.jpg)

## Reference
1. [Dive into Deep Learning](https://github.com/d2l-ai/d2l-zh/blob/master/chapter_computer-vision/ssd.md)
2. [An implementation of YOLOv3](https://github.com/bubbliiiing/yolo3-pytorch)
