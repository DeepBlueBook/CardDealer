import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import os


def parser():
    parser = argparse.ArgumentParser(description='For Plotting Curves')
    parser.add_argument(dest='file', type=str, help='file path to log file')
    parser.add_argument('--save_path', type=str,
                        default='./loss.png', help='file path to save image')
    parser.add_argument('--realtime', action='store_false',
                        help='whether to plot in realtime (defaual True)')
    args = parser.parse_args()
    return args


def plot(data, num, color='r', label=''):
    # plt.subplot(num)
    plt.plot(range(len(data)), data, color=color, label=label)
    plt.xlabel('Iterations')
    plt.ylabel('Training Loss')
    # plt.ylim(0,0.00025)
    # plt.xticks(range(0,4500,1000))


def read_from_file(args):
    data = []
    with open(args.file, 'r') as f:
        for i in f.readlines():
            i = i.strip('\n')
            data.append(i.split(','))
    return np.array(data)


def real_time_plot(args):
    size_pre = 0
    while True:
        size = os.stat(args.file).st_size
        if size != size_pre:
            plt.clf()
            data = read_from_file(args)
            train_loss = np.array(data[:, 0]).astype(np.float32)
            # train_loss = data[:, 1]
            plot(train_loss, 111, color='g', label='Loss')
            plt.legend()
            plt.tight_layout()
            size_pre = os.stat(args.file).st_size
            plt.pause(0.01)
        time.sleep(5)


def save(args):
    data = read_from_file(args)
    train_loss = np.array(data[:, 0]).astype(np.float32)
    # train_loss = data[:, 1]
    plot(train_loss, 111, color='g', label='Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.save_path)


if __name__ == '__main__':
    args = parser()
    if not os.path.exists(args.file):
        raise OSError('File not found at: {}'.format(args.file))
    if args.realtime:
        real_time_plot(args)
    else:
        save(args)
