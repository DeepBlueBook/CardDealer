# A PyTorch implementation of a YOLO v3 card Detector

This repository contains code for a card detector based on [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf), implementedin PyTorch. The code is based on the official code of [YOLO v3](https://github.com/pjreddie/darknet), as well as a PyTorch 
port of the original code, https://github.com/marvis/pytorch-caffe-darknet-convert

### 参考
[Implement YOLO v3 from scratch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)

## Requirements
1. Python 3.5
2. OpenCV
3. PyTorch 0.4

Using PyTorch 0.3 will break the detector.

## Running the training code

Please set some params. Check 
```
python train.py --help
```
