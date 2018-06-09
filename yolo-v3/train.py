# encoding: utf-8

from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from darknet import Darknet
from dataset import CardData
#from util import read_truths_args, read_data_cfg
from loss import V3Loss

import os
import argparse
from math import log10
import cv2
import glob


def parser():
    # Training settings
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--cfg", dest='cfg', help="config file",
                        default="cfg/yolo-v3-card.cfg", type=str)
    parser.add_argument("--weight", dest='weight', help="weight file",
                        default=None, type=str)
    parser.add_argument("--save", dest='save', help="save directory",
                        default="backup/", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=int)
    parser.add_argument('--batchsize', type=int, default=32,
                        help='training batch size')
    parser.add_argument('--nEpochs', type=int, default=100,
                        help='number of epochs to train for')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='a period of save checkpoint')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning Rate. Default=0.01')
    parser.add_argument('--lr_decay', type=int, default=5,
                        help='Dividing Learning Rate with the number. Default=5')
    parser.add_argument('--lr_step', type=int, default=30,
                        help='A period of deviding lr. Default=30')
    parser.add_argument('--cpu', action='store_true', help='use cuda?')
    parser.add_argument('--seed', type=int, default=123,
                        help='random seed to use. Default=123')
    parser.add_argument('--data_path', type=str, default="../dataset_generator/src/card_templates",
                        help='path to image dataset')
    args = parser.parse_args()
    return args


def main(args):
    # Device setting
    if not args.cpu and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    torch.manual_seed(args.seed)
    CUDA = not args.cpu
    device = torch.device("cpu" if args.cpu else "cuda")

    # Building model
    print('===> Building model')
    model = Darknet(args.cfg, CUDA).to(device)
    if args.weight:
        model.load_weights(args.weight)
    model.net_info["height"] = args.reso

    # Preparing for parse to loss func
    scale_anchors = []  # selection anchors in the respective scales
    for x in model.blocks:
        if x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [anchors[i] for k in mask for i in (2*k, 2*k+1)]
            scale_anchors.append(anchors)
            nC = int(x['classes'])
    args.reso = int(args.reso)
    strides = [32//(x+1) for x in range(len(scale_anchors))]
    resos = [args.reso//x for x in strides]
    num_boxes = sum([3*(x**2) for x in resos])

    nA = len(anchors) // 2
    print("num_classes={}, anchors={}, num_anchors={}, resolution={}, strides={}, device={}".format(
        nC, scale_anchors, nA, args.reso, strides, device))

    # Loss Function and Optimizer
    criterion = V3Loss(num_classes=nC, anchors=scale_anchors, num_anchors=nA,
                       resolution=args.reso, strides=strides, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Loading data
    print('===> Loading datasets')
    image_file_names = glob.glob(args.data_path + '/*')
    image_files = []
    for n in image_file_names:
        image_files.append(cv2.imread(n, 1))

    dataset = CardData(
        image_size=[args.reso, args.reso],
        image_list=image_files,
        image_classid_list=image_file_names,
        transform=transforms.Compose([
            transforms.ToTensor()]),
        data_size=args.batchsize*100,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batchsize, shuffle=False,
        num_workers=8, pin_memory=False)

    print("Batch size: {}, input_dim: {}".format(
        args.batchsize, [args.reso, args.reso]))

    # train and test
    for epoch in range(1, args.nEpochs + 1):
        train(model, optimizer, dataloader, criterion, device, epoch, args)
        # test(model, dataloader, criterion, device, epoch, args)


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr * ((1.0/args.lr_decay) ** (epoch // args.lr_step))
    # print("===> Epoch {} lr: {}".format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(model, optimizer, dataloader, criterion, device, epoch, args):
    lr = adjust_learning_rate(args, optimizer, epoch)
    epoch_loss = 0
    for iteration, batch in enumerate(dataloader, 1):
        input = batch[0].to(device)
        target = torch.tensor(batch[1]).to(device)

        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("\r===> Epoch[{}]({}/{}): lr: {} Loss: {:.3f}".format(epoch,
                                                                    iteration, len(dataloader), lr, loss.item()), end='')
        with open(os.path.join(args.save, 'loss.txt'), 'a') as f:
            f.write('{}\n'.format(loss.item()))
    print("\n ==> Epoch {} Complete: Avg. Loss: {:.4f}".format(
        epoch, epoch_loss / len(dataloader)))
    if (epoch+1) % args.save_interval == 0:
        print('save weights to %s/%06d.weights' % (args.save, epoch+1))
        model.seen = (epoch + 1) * len(dataloader)
        model.save_weights('%s/%06d.weights' % (args.save, epoch+1))
    


def test(model, dataloader, criterion, device, epoch, args):
    avg_loss = 0
    with torch.no_grad():
        for iteration, batch in enumerate(dataloader, 1):
            input = batch[0].to(device)
            target = torch.tensor(batch[1]).to(device)

            loss = criterion(model(input), target)
            avg_loss += loss
            if iteration > 100:
                break
    print("===>Epoch: {} Avg. Loss: {:.4f} dB\n".format(epoch, avg_loss / 100))


if __name__ == "__main__":
    args = parser()
    assert args.reso % 32 == 0
    assert args.reso > 32
    if not os.path.exists(args.save):
        os.mkdir(args.save)
    if not os.path.exists(args.data_path):
        msg = 'Folder not found at: {}\nSpecify correct path with argument --data_path'.format(args.data_path)
        raise OSError(msg)
    if os.path.exists(os.path.join(args.save, 'loss.txt')):
        os.remove(os.path.join(args.save, 'loss.txt'))
    main(args)
