## Tiny-SSD
An implementation of SSD (Code is refenced from Dive into Deep Learning)

## Usage
1. Download voc2007 or voc2012 or voc2077(my small toy face detection dataset)
2. Put them in the right directory
3. Run voc_annotation to generate divided dataset(in VOCdevkit\VOC20..\ImageSets\Main) and dataset labels(in code's main directory)
4. Run train.py to train a simple model(will not consume too many time). It will save *.pt weights in model_date/ automatically every epoch
5. Then you can run inference to get object detection result from a picture or your pc camera
