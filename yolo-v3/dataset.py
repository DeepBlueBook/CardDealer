#!/usr/bin/python
# encoding: utf-8

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
#from util import read_truths_args, read_truths
#from image import *

from scipy.misc import imread, imsave
# from random import *
import cv2


class listDataset(Dataset):

    def __init__(self, root, shape=None, shuffle=True, transform=None, target_transform=None, train=False, seen=0, batch_size=64, num_workers=4):
        with open(root, 'r') as file:
            self.lines = file.readlines()

        if shuffle:
            random.shuffle(self.lines)

        self.nSamples = len(self.lines)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        imgpath = self.lines[index].rstrip()

        if self.train and index % 64 == 0:
            if self.seen < 4000*64:
                width = 13*32
                self.shape = (width, width)
            elif self.seen < 8000*64:
                width = (random.randint(0, 3) + 13)*32
                self.shape = (width, width)
            elif self.seen < 12000*64:
                width = (random.randint(0, 5) + 12)*32
                self.shape = (width, width)
            elif self.seen < 16000*64:
                width = (random.randint(0, 7) + 11)*32
                self.shape = (width, width)
            else:  # self.seen < 20000*64:
                width = (random.randint(0, 9) + 10)*32
                self.shape = (width, width)

        if self.train:
            jitter = 0.2
            hue = 0.1
            saturation = 1.5
            exposure = 1.5

            img, label = load_data_detection(
                imgpath, self.shape, jitter, hue, saturation, exposure)
            label = torch.from_numpy(label.astype('float32'))

        else:
            img = Image.open(imgpath).convert('RGB')
            if self.shape:
                img = img.resize(self.shape)

            labpath = imgpath.replace('images', 'labels').replace(
                'JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')
            label = torch.zeros(50*5)

            # if os.path.getsize(labpath):
            #tmp = torch.from_numpy(np.loadtxt(labpath))
            try:
                tmp = torch.from_numpy(read_truths_args(
                    labpath, 8.0/img.width).astype('float32'))
            except Exception:
                tmp = torch.zeros(1, 5)
            #tmp = torch.from_numpy(read_truths(labpath))
            tmp = tmp.view(-1)
            tsz = tmp.numel()
            #print('labpath = %s , tsz = %d' % (labpath, tsz))
            if tsz > 50*5:
                label = tmp[0:50*5]
            elif tsz > 0:
                label[0:tsz] = tmp

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        self.seen = self.seen + self.num_workers
        return (img, label)


class DataGenerator():
    def __init__(self, batch_size, image_size, image_list, image_classid_list, max_num=1, epoch_len=1000):
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_channel = image_list[0].shape[2]

        self.image_list = image_list
        self.image_classid_list = image_classid_list

        self.max_num = max_num
        self.epoch_len = epoch_len

    def generate_image(self):
        idx = np.random.randint(0, len(self.image_list))

        src_image = self.image_list[idx]

        class_id = idx

        # decide center
        bx = np.random.random()
        by = np.random.random()

        center = [(int)(bx*self.image_size[0]), (int)(by*self.image_size[1])]

        # decide scale
        scale = 0.4
        bw = src_image.shape[1]*scale/self.image_size[0]
        bh = src_image.shape[0]*scale/self.image_size[1]

        resized_image = cv2.resize(src_image, ((int)(
            scale * src_image.shape[1]), (int)(scale * src_image.shape[0])))

        canvas = np.zeros(
            (self.image_size[0], self.image_size[1]), dtype=np.uint8)

        T = np.array([[1.0, 0.0, 1.0*(int)(bx*canvas.shape[0]) - resized_image.shape[1]/2],
                      [0.0, 1.0, 1.0*(int)(by*canvas.shape[1]) -
                       resized_image.shape[0]/2],
                      [0.0, 0.0, 1.0]])
        canvas = cv2.warpPerspective(
            resized_image, T, (canvas.shape[0], canvas.shape[1]))

        return canvas, class_id, bx, by, bw, bh

    def generate(self):
        while True:
            X = np.zeros((self.batch_size, self.image_channel,
                          self.image_size[0], self.image_size[1]))
            Y = np.zeros((self.batch_size, 5*50))
            for i in range(self.batch_size):
                img, cls, bx, by, bw, bh = self.generate_image()
                X[i] = img.transpose(2, 0, 1)
                # Y[i] = np.c_[cls, bx, by, bw, bh]  # np.r_かも
                Y[i, :5] = np.array([cls, bx, by, bw, bh])  # np.r_かも
            yield X, Y

    def __len__(self):
        return self.epoch_len


if __name__ == "__main__":
    image_data_path = "/ceph/rmurase/Datasets/card_templates/"
    _image_files = os.listdir(image_data_path)
    image_file_names = list(
        map(lambda image_file_name: image_data_path + image_file_name, _image_files))
    print(image_file_names)

    image_files = []

    for n in image_file_names:
        image_files.append(cv2.imread(n, 1))
    print(image_files[0])
    dataloader = DataGenerator(
        2, [512, 512], image_files, image_file_names, max_num=1)
    print("len(dataloader)", len(dataloader))
    for iter, (input, target) in enumerate(dataloader.generate()):
        if iter == 3:
            print(input.shape, target.shape)
            src = input[0].transpose(1, 2, 0)
            print(src)
            cls, bx, by, bw, bh = target[0][:5]
            im_size = src.shape[0]
            import matplotlib.patches as patches
            import matplotlib.pyplot as plt
            bb = np.array([bx, by, bw, bh]).astype(np.int)

            # Create figure and axes
            fig, ax = plt.subplots(1)

            # Display the image
            ax.imshow(src)
            rect = patches.Rectangle(
                (bb[0]-bb[2]/2, bb[1]-bb[3]/2), bb[2], bb[3], linewidth=3, edgecolor='r')
            ax.add_patch(rect)
            plt.savefig("bb_test.jpg")

            #t_l = ((bx - bw/2)*im_size,(by - bh/2)*im_size)
            #b_r = ((bx + bw/2)*im_size,(by + bh/2)*im_size)
            #t_l = [int(x) for x in t_l]
            #b_r = [int(x) for x in b_r]
            #cv2.rectangle(src, t_l, b_r, (0,255,0),3)

            cv2.imwrite("test0.jpg", input[0].transpose(1, 2, 0))
            cv2.imwrite("test1.jpg", input[1].transpose(1, 2, 0))
            break
    #data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
