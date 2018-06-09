from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from darknet import Darknet
from dataset import  DataGenerator
#from util import read_truths_args, read_data_cfg
from loss import V3Loss

import os
import argparse
from math import log10
import cv2

### Training settings
parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
parser.add_argument("--cfg", dest = 'cfg', help = 
                    "config file",
                    default = "cfg/yolo-v3-card.cfg", type = str)
parser.add_argument("--weight", dest = 'weight', help = 
                    "weight file",
                    default = None, type = str)
parser.add_argument("--save", dest = 'save', help = 
                    "save directory",
                    default = "backup/", type = str)
parser.add_argument("--reso", dest = 'reso', help = 
                    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                    default = "416", type = str)
parser.add_argument('--batchsize', type=int, default=32, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--save_interval', type=int, default=10, help='a period of save checkpoint')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.01')
parser.add_argument('--lr_decay', type=int, default=5, help='Dividing Learning Rate with the number. Default=5')
parser.add_argument('--lr_step', type=int, default=30, help='A period of deviding lr. Default=30')
parser.add_argument('--cpu', action='store_true', help='use cuda?')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
args = parser.parse_args()

assert args.reso % 32 == 0 
assert args.reso > 32 
if not os.path.exists(args.save):
    os.mkdir(args.save)

### Device setting
if not args.cpu and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")
torch.manual_seed(args.seed)
CUDA = not args.cpu
device = torch.device("cpu" if args.cpu else "cuda")

### Building model
print('===> Building model')
model = Darknet(args.cfg, CUDA).to(device)
if args.weight:
    model.load_weights(args.weight)
model.net_info["height"] = args.reso
       
### Preparing for parse to loss func
scale_anchors = []  ### selection anchors in the respective scales
for x in model.blocks:
    if x["type"] == "yolo":
        mask = x["mask"].split(",")
        mask = [int(x) for x in mask]
        anchors = x["anchors"].split(",")
        anchors = [int(a) for a in anchors]
        anchors = [anchors[i] for k in mask for i in (2*k, 2*k+1) ]
        scale_anchors.append(anchors)
        nC = int(x['classes'])
args.reso = int(args.reso)
strides = [32//(x+1) for x in range(len(scale_anchors))]
resos = [args.reso//x for x in strides]
num_boxes = sum([3*(x**2) for x in resos])  

nA= len(anchors) // 2
print("num_classes={}, anchors={}, num_anchors={}, resolution={}, strides={}, device={}".format(nC,scale_anchors,nA,args.reso,strides, device))

# Loss Function and Optimizer
criterion = V3Loss(num_classes=nC, anchors=scale_anchors, num_anchors=nA, resolution=args.reso, strides=strides, device=device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Loading data
print('===> Loading datasets')
image_data_path = "/ceph/rmurase/Datasets/card_templates/"
_image_files = os.listdir(image_data_path)
image_file_names = list(map(lambda image_file_name : image_data_path + image_file_name , _image_files))
image_files = []
for n in image_file_names:
        image_files.append(cv2.imread( n,1))
dataloader = DataGenerator(args.batchsize, [args.reso, args.reso], image_files, image_file_names, max_num = 1)  # default args.reso=416
print("Batch size{}, input_dim{}".format(args.batchsize, [args.reso, args.reso]))

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * ((1.0/args.lr_decay) ** (epoch // args.lr_step))
    print("===> Epoch {} lr: {}".format(epoch,lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(epoch):
    lr = adjust_learning_rate(optimizer, epoch)
    epoch_loss = 0
    for iteration, batch in enumerate(dataloader.generate(), 1):
        input, target = torch.tensor(batch[0],requires_grad=True,dtype=torch.float).to(device), torch.tensor(batch[1],dtype=torch.float).to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("===> Epoch[{}]({}/{}): lr{} Loss: {:.4f}".format(epoch, iteration, len(dataloader), lr, loss.item()))
        if iteration == len(dataloader):
            break
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(dataloader)))
    if (epoch+1) % args.save_interval == 0:
        print('save weights to %s/%06d.weights' % (args.save, epoch+1))
        model.seen = (epoch + 1) * len(dataloader)
        model.save_weights('%s/%06d.weights' % (args.save, epoch+1))


def test(epoch):
    avg_loss = 0
    with torch.no_grad():
        for iteration, batch in enumerate(dataloader.generate(), 1):
            input, target = torch.tensor(batch[0],dtype=torch.float).to(device), torch.tensor(batch[1],dtype=torch.float).to(device)

            prediction = model(input)
            mse = criterion(prediction, target)
            loss = 10 * log10(1 / mse.item())
            avg_loss += loss
            if iteration > 100:
                break
    print("===>Epoch: {} Avg. Loss: {:.4f} dB".format(epoch, avg_loss / 100))


for epoch in range(1, args.nEpochs + 1):
    train(epoch)
    test(epoch)