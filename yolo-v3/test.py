# encoding: utf-8

from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from darknet import Darknet
from dataset import CardData
from loss import V3Loss

import os
import argparse
import cv2
import glob


def parser():
    # Test settings
    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--weight", dest='weight', help="weight file",
                        type=str, default='backup/model_best.pth.tar')
    parser.add_argument("--cfg", dest='cfg', help="config file",
                        default="cfg/yolo-v3-card.cfg", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default=416, type=int)
    parser.add_argument('--batchsize', type=int, default=1,
                        help='training batch size')
    parser.add_argument('--cpu', action='store_true', help='use cuda?')
    parser.add_argument('--data_path', type=str, default="/ceph/rmurase/Datasets/card_templates/",
                        help='path to image dataset')
    args = parser.parse_args()
    return args


def main(args):
    # Device setting
    if not args.cpu and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    CUDA = not args.cpu
    device = torch.device("cpu" if args.cpu else "cuda")

    # Building model
    print('===> Building model')
    model = Darknet(args.cfg, CUDA).to(device)
    # model.load_weights(args.weight)
    model.net_info["height"] = args.reso
    # Load saved model weight
    checkpoint = torch.load(args.weight)
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    print("===> Loaded checkpoint '{}' (epoch {})"
            .format(args.weight, checkpoint['epoch']))

    # Preparing for parse to loss func
    scale_anchors = []  # selection anchors in the respective scales
    for x in model.blocks:
        if x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [anchors[i] for k in mask for i in (2 * k, 2 * k + 1)]
            scale_anchors.append(anchors)
            nC = int(x['classes'])
    args.reso = int(args.reso)
    strides = [32 // (x + 1) for x in range(len(scale_anchors))]
    resos = [args.reso // x for x in strides]
    num_boxes = sum([3 * (x**2) for x in resos])

    nA = len(anchors) // 2
    print("num_classes={}, anchors={}, num_anchors={}, resolution={}, strides={}, device={}".format(
        nC, scale_anchors, nA, args.reso, strides, device))

    # Loss Function and Optimizer
    criterion = V3Loss(num_classes=nC, anchors=scale_anchors, num_anchors=nA,
                       resolution=args.reso, strides=strides, device=device)

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
        data_size=args.batchsize * 100,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batchsize, shuffle=False,
        num_workers=8, pin_memory=False)

    print("Batch size: {}, input_dim: {}".format(
        args.batchsize, [args.reso, args.reso]))

    # test
    test(model, dataloader, criterion, device, args)


def test(model, dataloader, criterion, device, args):
    model.eval()
    epoch_loss = 0
    for iteration, batch in enumerate(dataloader, 1):
        input = batch[0].to(device)
        target = torch.tensor(batch[1]).to(device)

        loss = criterion(model(input), target)
        epoch_loss += loss.item()

        print("\r===> Testing {}/{}".format(iteration, len(dataloader)), end='')

    print("\n ==> Complete: Avg. Loss: {:.3f}".format(
        epoch_loss / len(dataloader)))


if __name__ == "__main__":
    args = parser()
    assert args.reso % 32 == 0
    assert args.reso > 32
    if not os.path.exists(args.data_path):
        msg = 'Folder not found at: {}\nSpecify correct path with argument --data_path'.format(
            args.data_path)
        raise OSError(msg)
    main(args)
