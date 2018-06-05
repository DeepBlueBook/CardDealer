### pytorch-yolo2
Convert https://pjreddie.com/darknet/yolo/ into pytorch, originally https://github.com/marvis/pytorch-caffe-darknet-convert

動作確認環境
* python3.5.5
* pytorch0.4.0
* torchvision0.2.1

### Downlo weight and config file
https://pjreddie.com/darknet/yolo/ を参照してダウンロードしてください．


##### Train The Model
if yolo-v2 full version
```
python train.py cfg/voc.data cfg/yolo-voc.cfg darknet19_448.conv.23
```
elif tiny-yolo-v2
```
python train.py cfg/voc.data cfg/tiny-yolo-voc.cfg darknet19_448.conv.23
```

### Demo
```
python detect.py cfg/tiny-yolo-voc.cfg weights/yolov2-tiny-voc.weights data/dog.jpg
```

なぜかこれだと動かないので調査中です．
```
python detect.py cfg/yolov2-voc.cfg weights/yolov2-voc.weights data/dog.jpg
```

---
#### License
MIT License (see LICENSE file).
