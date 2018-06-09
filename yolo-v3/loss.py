import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from util import *
from bbox import *
import argparse


def build_targets(pred_boxes, target, anchors, num_anchors, num_classes, nH, nW, noobject_scale, object_scale, sil_thresh, seen, device):
    nB = target.size(0)  # num of batchess
    nA = num_anchors
    nC = num_classes
    # sually anchors is pair of integer, so anchor_step is usually 2.
    anchor_step = len(anchors) // num_anchors
    # confusion mask, if no obj exist,  5
    conf_mask = torch.ones(nB, nA * nH * nW).to(device) * noobject_scale
    coord_mask = torch.zeros(nB, nA * nH * nW).to(device)
    cls_mask = torch.zeros(nB, nA * nH * nW).to(device)
    tx = torch.zeros(nB, nA * nH * nW).to(device)
    ty = torch.zeros(nB, nA * nH * nW).to(device)
    tw = torch.zeros(nB, nA * nH * nW).to(device)
    th = torch.zeros(nB, nA * nH * nW).to(device)
    tconf = torch.zeros(nB, nA * nH * nW).to(device)
    tcls = torch.zeros(nB, nA * nH * nW, nC).to(device)

    nAnchors = nA * nH * nW
    nPixels = nH * nW
    for b in range(nB):
        cur_pred_boxes = pred_boxes[b, 0:nAnchors]
        cur_ious = torch.zeros(nAnchors).to(device)

        #  If target image has n objects, target vector is like [cl0, tx0, ty0, tw0, th0, cl1, tx1, ty1, ...... , cln, txn, tyn, twn, thn, 0, 0, 0, 0, 0, 0, 0, .....]  (shape is [50])
        #  cl : class number in the definition of the config file whose name is like "*.name".
        #  tx, ty, tw, th: bounding box center, width and height, which are normalized by the image size.
        for t in range(50):
            if target[b][t * 5 + 1] == 0:
                break
            gx = target[b][t * 5 + 1] * nW
            # print(gx,cur_pred_boxes)
            gy = target[b][t * 5 + 2] * nH
            gw = target[b][t * 5 + 3] * nW
            gh = target[b][t * 5 + 4] * nH
            cur_gt_boxes = torch.FloatTensor(
                [gx, gy, gw, gh]).repeat(nAnchors, 1).to(device)
            # print(cur_gt_boxes)
            cur_ious = torch.max(cur_ious, bbox_iou(
                cur_pred_boxes, cur_gt_boxes, xywh=True))
        #print("cur_ious",cur_ious.shape, "conf_mask", conf_mask.shape)
        conf_mask[b][cur_ious > sil_thresh] = 0
    if seen < 12800:
        if anchor_step == 4:
            tx = torch.FloatTensor(anchors).view(nA, anchor_step).index_select(
                1, torch.LongTensor([2])).view(1, nA, 1, 1).repeat(nB, 1, nH, nW).to(device)
            ty = torch.FloatTensor(anchors).view(num_anchors, anchor_step).index_select(
                1, torch.LongTensor([2])).view(1, nA, 1, 1).repeat(nB, 1, nH, nW).to(device)
        else:
            tx.fill_(0.5)
            ty.fill_(0.5)
        tw.zero_()
        th.zero_()
        coord_mask.fill_(1)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(50):
            if target[b][t * 5 + 1] == 0:
                break
            nGT = nGT + 1
            best_iou = 0.0
            best_n = -1  # best_anchor_number
            min_dist = 10000  # min distance
            gx = target[b][t * 5 + 1] * nW
            gy = target[b][t * 5 + 2] * nH
            gi = int(gx)    # (grid_index)x(img_size/stride)
            gj = int(gy)    #
            gw = target[b][t * 5 + 3] * nW
            gh = target[b][t * 5 + 4] * nH
            gt_box = torch.FloatTensor([0, 0, gw, gh]).view(1, 4).to(device)
            for n in range(nA):
                aw = anchors[anchor_step * n]
                ah = anchors[anchor_step * n + 1]
                anchor_box = torch.FloatTensor(
                    [0, 0, aw, ah]).view(1, 4).to(device)
                iou = bbox_iou(anchor_box, gt_box, xywh=True)
                if anchor_step == 4:
                    ax = anchors[anchor_step * n + 2]
                    ay = anchors[anchor_step * n + 3]
                    dist = pow(((gi + ax) - gx), 2) + pow(((gj + ay) - gy), 2)
                if iou > best_iou:
                    best_iou = iou
                    best_n = n
                elif anchor_step == 4 and iou == best_iou and dist < min_dist:
                    best_iou = iou
                    best_n = n
                    min_dist = dist

            gt_box = torch.FloatTensor([gx, gy, gw, gh]).view(1, 4).to(device)
            best_idx = nA * (nW * gi + gj) + best_n
            pred_box = pred_boxes[b, best_idx].view(1, 4)
            tcl_idx = int(target[b][t * 5])

            coord_mask[b, best_idx] = 1
            cls_mask[b, best_idx] = 1
            conf_mask[b, best_idx] = object_scale

            # (tx, ty, tw, th) == (bx, by, bw, bh) in the paper of yolo-v3
            tx[b, best_idx] = target[b][t * 5 + 1]
            ty[b, best_idx] = target[b][t * 5 + 2]
            tw[b, best_idx] = target[b][t * 5 + 3]
            th[b, best_idx] = target[b][t * 5 + 4]
            #tx[b, best_idx] = target[b][t*5+1] * nW - gi
            #ty[b, best_idx] = target[b][t*5+2] * nH - gj
            # tw[b, best_idx] = math.log(gw/anchors[anchor_step*best_n])  # ???
            # th[b, best_idx] = math.log(gh/anchors[anchor_step*best_n+1]) # ???
            iou = bbox_iou(gt_box, pred_box)  # best_iou
            tconf[b, best_idx] = iou
            tcls[b, best_idx, tcl_idx] = torch.FloatTensor([1]).to(device)
            if iou > 0.5:
                nCorrect = nCorrect + 1
    return nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls


class V3Loss(nn.Module):
    def __init__(self, num_classes=0, anchors=[], num_anchors=1, resolution=416, strides=[], device=torch.device("cpu")):
        super(V3Loss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = 3
        self.anchor_step = 2
        self.coord_scale = 1
        self.noobject_scale = 1
        self.object_scale = 5
        self.class_scale = 1
        self.thresh = 0.6
        self.seen = 0
        self.reso = resolution
        strides.sort(reverse=True)  # bigger first
        self.strides = strides
        self.device = device
        self.output_split = 0
        self.losses = []

    def forward(self, output, target):
        # output : BxAs*(4+1+num_classes)*H*W
        t0 = time.time()
        nB = output.data.size(0)
        nA = self.num_anchors
        nC = self.num_classes
        self.output_split = 0
        self.losses = []

        for i, stride in enumerate(self.strides):
            nH = self.reso // stride
            nW = self.reso // stride
            nAnchors = nH * nW * nA
            anchors = self.anchors[i]

            # Normarize
            scale_split_output = output[:,
                                        self.output_split: self.output_split + nAnchors, 0:4] / self.reso
            conf = output[:,
                          self.output_split: self.output_split + nAnchors, 5]
            cls = output[:, self.output_split: self.output_split + nAnchors, 5:]
            self.output_split = self.output_split + nAnchors

            vals = {
                "output": scale_split_output,
                "target": target,
                "anchors": anchors,
                "nA": nA,
                "nC": nC,
                "nH": nH,
                "nW": nW,
                "noobject_scale": 5, "object_scale": 1, "thresh": 0.6, "seen": False, "device": self.device
            }
            nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls = build_targets(
                vals["output"], vals["target"], vals["anchors"], vals["nA"], vals["nC"], vals["nH"], vals["nW"],
                vals["noobject_scale"], vals["object_scale"], vals["thresh"], vals["seen"], vals["device"])

            loss_x = self.coord_scale * nn.MSELoss(size_average=False)(
                scale_split_output[:, :, 0] * coord_mask, tx * coord_mask) / 2.0
            loss_y = self.coord_scale * nn.MSELoss(size_average=False)(
                scale_split_output[:, :, 1] * coord_mask, ty * coord_mask) / 2.0
            loss_w = self.coord_scale * nn.MSELoss(size_average=False)(
                scale_split_output[:, :, 2] * coord_mask, tw * coord_mask) / 2.0
            loss_h = self.coord_scale * nn.MSELoss(size_average=False)(
                scale_split_output[:, :, 3] * coord_mask, th * coord_mask) / 2.0
            loss_conf = nn.MSELoss(size_average=False)(
                conf * conf_mask, tconf * conf_mask) / 2.0
            loss_cls = self.class_scale * \
                nn.BCELoss(size_average=False)(cls, tcls)   # Todo
            loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls
            # print("loss_x {} loss_y {} loss_w {} loss_h {} loss_conf {} loss_cls {}".format(
            #     loss_x, loss_y, loss_w, loss_h, loss_conf, loss_cls))
            self.losses.append(loss)
        return sum(self.losses)


if __name__ == "__main__":
    from darknet import Darknet
    from detect import get_test_input

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--cfg", dest='cfg', help="config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--reso", dest='reso', help="Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    parser.add_argument("--tiny", action="store_true",
                        help="Using yolov3-tiny")
    parser.add_argument("--scales", dest="scales", help="Scales to use for detection",
                        default="1,2,3", type=str)
    args = parser.parse_args()
    print(args)

    CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if CUDA else "cpu")

    scales = args.scales
    scales = [int(x) for x in scales.split(',')]

    scales_indices = []
    args.reso = int(args.reso)
    strides = [8, 16, 32]
    resos = [args.reso // x for x in strides]
    num_boxes = sum([3 * (x**2) for x in resos])

    for scale in scales:
        li = list(range((scale - 1) * num_boxes // 3, scale * num_boxes // 3))
        scales_indices.extend(li)

    if CUDA:
        model = Darknet(args.cfg)
        model.net_info["height"] = args.reso
        inp_dim = int(model.net_info["height"])
        assert inp_dim % 32 == 0
        assert inp_dim > 32
        model.cuda()
        model.eval()
        _output = model(get_test_input(inp_dim, CUDA), CUDA)
        print(model.blocks)
        #_output = convert2cpu(_output)
    else:
        inp_dim = args.reso
        _output = torch.FloatTensor(np.ones((1, num_boxes, 85)))

    _test_target = np.zeros((1, 5 * 50))
    _test_target[:, 0] = 1
    _test_target[:, 1:5] = 0.2
    _test_target[:, 5] = 5
    _test_target[:, 6:10] = 0.1
    test_target = torch.FloatTensor(_test_target).to(device)
    anchors = [116, 90,  156, 198,  373, 326]
    #loss = V3Loss(num_classes=10, anchors=anchors, num_anchors=3)
    #lossval = loss(test_output,test_target)

    # use prediction when stride 32
    nH = args.reso // 32
    nW = args.reso // 32
    nA = len(anchors) // 2
    nC = 80

    if False:
        test_output = _output[:, 0:nH * nW * nA, 0:4]
        vals = {
            "output": test_output,
            "target": test_target,
            "anchors": anchors,
            "nA": nA,
            "nC": nC,
            "nH": nH,
            "nW": nW,
            "noobject_scale": 5, "object_scale": 1, "thresh": 0.6, "seen": False, "device": device
        }
        nGT, nCorrect, coord_mask, conf_mask, cls_mask, tx, ty, tw, th, tconf, tcls = build_targets(
            vals["output"], vals["target"], vals["anchors"], vals["nA"], vals["nC"], vals["nH"], vals["nW"], vals["noobject_scale"], vals["object_scale"], vals["thresh"], vals["seen"], vals["device"])

    if True:
        test_output = _output
        v3loss = V3Loss(num_classes=80, anchors=anchors, num_anchors=3,
                        resolution=args.reso, strides=strides, device=device)
        lossval = v3loss(test_output, test_target)
        print(lossval)
